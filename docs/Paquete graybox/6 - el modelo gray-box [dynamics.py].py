# =============================================================================
#  MODELO DE DINAMICA  ẋ = f_θ(x, P, Q)   (Neural ODE gray-box)
# =============================================================================
#
#  A diferencia de la PINN (que mapea t -> [I,E] para un estimulo FIJO), este
#  modelo es de ESTADOS: dado el estado actual [I,E] y el control [P,Q], devuelve
#  la derivada [dI/dt, dE/dt]. Se integra paso a paso -> se puede meter en un lazo
#  de control donde P,Q se generan online (lo que la PINN no permite).
#
#  Es GRAY-BOX: backbone con la estructura EXACTA de Wilson-Cowan (misma que
#  losses.py / model.py) + una correccion neuronal g_φ opcional. Conservar la
#  estructura WC es REQUISITO para que el controlador IMC (feedback linearization)
#  pueda cancelar el acoplamiento. Ver docs/plan_planta_neural.html.
#
#  Variantes (segun los flags):
#    V0 (white-box): use_correction=False, learnable_weights=False
#        -> son las ecuaciones WC con los pesos dados (planta de referencia).
#    V1 (gray-box):  use_correction=True
#        -> WC + MLP que captura lo que el backbone no modela (la parte aprendida).
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrayBoxWC(nn.Module):
    WEIGHTS = ("wEE", "wEI", "wIE", "wII")
    # Parametros "fisicos" que tambien se pueden identificar (ademas de los pesos).
    # ke,ki NO estan aca: no son libres, se derivan de ae,thetae / ai,thetai.
    EXTRA = ("te", "ti", "ae", "ai", "thetae", "thetai")

    def __init__(
        self,
        fixed: dict,                 # te,ti,ae,ai,thetae,thetai,ke,ki (floats)
        w_init: dict,                # wEE,wEI,wIE,wII (p.ej. los identificados por la PINN)
        learnable_weights: bool = False,  # True -> los 4 pesos se siguen ajustando
        use_correction: bool = False,     # True -> agrega la correccion neuronal g_φ (V1)
        learnable_params: bool = False,   # True -> tambien aprende te,ti,ae,ai,thetae,thetai
        hidden: int = 32,
        correction_inputs: str = "xpq",   # "xpq" -> g(I,E,P,Q) | "x" -> g(I,E)
        structured: bool = False,         # True -> correccion FISICA en vez de red
    ) -> None:
        super().__init__()

        # --- CORRECCION ESTRUCTURADA (alternativa a la red neuronal) ----------
        #  En vez de un MLP generico de ~1000 pesos, se agregan TRES parametros
        #  con significado fisico, que son la forma que uno sospecha que tiene el
        #  hueco:
        #     r_i, r_e : refractariedad, el factor (1 - r*x) del Wilson-Cowan
        #                original de 1972 que la forma reducida descarta.
        #     sat      : nivel de saturacion del actuador (la opsina no entrega
        #                corriente infinita).
        #
        #  Por que conviene: es identificable (3 numeros, no una red),
        #  interpretable, y sobre todo la cancelacion del controlador vuelve a
        #  ser EXACTA Y EXPLICITA, porque se conoce la forma analitica y se
        #  puede invertir. Una correccion de caja negra no permite eso (se midio:
        #  arruina un canal del lazo cerrado aunque ajuste bien).
        #
        #  Lo que esta correccion NO puede capturar: el RETARDO del actuador, que
        #  tiene estado propio. Un modelo de estados sin memoria no lo alcanza.
        #  Es una limitacion honesta y esperada.
        #
        #  Arranque: r ~ 0 y saturacion casi nula -> empieza siendo WC puro.
        #
        #  PARAMETRIZACION — importa mucho, y la primera version estaba mal.
        #  Si se usa softplus con el crudo muy negativo (para que r arranque en
        #  ~0), la derivada del softplus vale ~0.0025 y el gradiente de r muere:
        #  el parametro no se mueve nunca. Lo mismo con `sat` grande: como
        #  sat*tanh(P/sat) ~ P para sat grande, la derivada respecto de sat es
        #  del orden de P³/sat² ~ 1/2500. Los dos arrancaban en zonas planas.
        #
        #  Por eso: (a) r se parametriza DIRECTO con clamp (derivada 1), y
        #  (b) la saturacion se parametriza por su INVERSA alpha = 1/sat, cuya
        #  derivada cerca de cero vale ~P³/3 en vez de ~P³/sat². El limite
        #  alpha -> 0 es el canal ideal (sin saturacion).
        self.structured = structured
        if structured:
            self.raw_r_i = nn.Parameter(torch.tensor(0.01))     # r_i, directo
            self.raw_r_e = nn.Parameter(torch.tensor(0.01))     # r_e, directo
            self.raw_alpha = nn.Parameter(torch.tensor(0.01))   # alpha = 1/sat

        # --- Parametros fisicos (te,ti,ae,ai,thetae,thetai) ---------------------
        #  Caso por defecto: son constantes conocidas -> buffers (no se entrenan).
        #  Caso "todo aprendible": se guardan crudos (softplus -> >0, como los pesos)
        #  y ademas ke,ki dejan de ser fijos: se recalculan en cada forward a partir
        #  de los ae,thetae / ai,thetai actuales, para que E=I=0 siga siendo reposo.
        self._learn_extra = learnable_params
        if learnable_params:
            for k in self.EXTRA:
                v = torch.tensor(float(fixed[k]))
                self.register_parameter(f"raw_{k}", nn.Parameter(torch.log(torch.expm1(v))))
        else:
            for k in ("te", "ti", "ae", "ai", "thetae", "thetai", "ke", "ki"):
                self.register_buffer(k, torch.tensor(float(fixed[k])))

        # Pesos sinapticos. Si son entrenables, se guardan "crudos" (softplus -> >0).
        w = torch.tensor([float(w_init[k]) for k in self.WEIGHTS])
        self._learn = learnable_weights
        if learnable_weights:
            self.raw_w = nn.Parameter(torch.log(torch.expm1(w)))   # inv_softplus
        else:
            self.register_buffer("w_fixed", w)

        # Correccion neuronal opcional. Se inicializa en ~0 (arranca = backbone WC).
        #
        # QUE ARGUMENTOS RECIBE g_φ — es una decision de diseno con consecuencias:
        #
        #   "xpq" -> g(I, E, P, Q).  Es un aproximador universal sobre EXACTAMENTE
        #       el mismo dominio que el backbone. Consecuencia: para cualquier θ
        #       equivocado existe un g que compensa el error -> los 10 parametros
        #       dejan de ser identificables. La red aprende a tapar parametros
        #       malos en vez de aprender la fisica que falta.
        #
        #   "x"   -> g(I, E).  Al no ver el estimulo, TODA la dependencia de P,Q
        #       queda forzada dentro del backbone. Como P,Q son conocidos y ricos,
        #       eso ancla la escala de los parametros. Ademas hace que el
        #       controlador IMC pueda cancelar la correccion de forma EXACTA y
        #       explicita (si g dependiera de P,Q la cancelacion seria implicita
        #       en P,Q y habria que resolver un punto fijo en cada paso).
        #
        # Ver docs/incertidumbre_dinamica_graybox.md, seccion 6.
        self.use_correction = use_correction
        if correction_inputs not in ("xpq", "x"):
            raise ValueError("correction_inputs debe ser 'xpq' o 'x'")
        self.correction_inputs = correction_inputs
        if use_correction:
            n_in = 4 if correction_inputs == "xpq" else 2
            self.g = nn.Sequential(
                nn.Linear(n_in, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 2),
            )
            for m in self.g:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(self.g[-1].weight)   # salida 0 al inicio

    # Pesos reales (positivos).
    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_w) if self._learn else self.w_fixed

    def weights_dict(self) -> dict[str, float]:
        w = self.weights().detach()
        return {k: float(w[i]) for i, k in enumerate(self.WEIGHTS)}

    # Valor actual de un parametro fisico (softplus si es aprendible, buffer si no).
    def _extra(self, k: str) -> torch.Tensor:
        return F.softplus(getattr(self, f"raw_{k}")) if self._learn_extra else getattr(self, k)

    def params_dict(self) -> dict[str, float]:
        """Todos los parametros identificados (pesos + fisicos si son aprendibles)."""
        d = self.weights_dict()
        if self._learn_extra:
            for k in self.EXTRA:
                d[k] = float(self._extra(k).detach())
        return d

    # -------------------------------------------------------------------------
    #  BACKBONE: solo la parte Wilson-Cowan, SIN la correccion neuronal.
    #  Se separa del forward para poder (a) medir cuanto aporta cada parte,
    #  (b) calcular las sensibilidades ∂f/∂θ, y (c) cancelar en el controlador.
    # -------------------------------------------------------------------------
    def backbone(self, x: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        I = x[..., 0:1]
        E = x[..., 1:2]
        P = torch.as_tensor(P, dtype=x.dtype, device=x.device) * torch.ones_like(I)
        Q = torch.as_tensor(Q, dtype=x.dtype, device=x.device) * torch.ones_like(I)

        # Correccion estructurada: saturacion del canal de actuacion.
        # Con alpha = 1/sat:  u_eff = tanh(alpha*u)/alpha  -> u  cuando alpha -> 0.
        if self.structured:
            alpha = torch.clamp(self.raw_alpha, 1e-3, 10.0)
            P = torch.tanh(alpha * P) / alpha
            Q = torch.tanh(alpha * Q) / alpha

        w = self.weights()
        wEE, wEI, wIE, wII = w[0], w[1], w[2], w[3]

        # Parametros fisicos (constantes o aprendidos, segun learnable_params).
        te, ti = self._extra("te"), self._extra("ti")
        ae, ai = self._extra("ae"), self._extra("ai")
        thetae, thetai = self._extra("thetae"), self._extra("thetai")
        # Offsets de reposo: fijos si no se aprende nada, o recalculados de los
        # ae,thetae / ai,thetai actuales para conservar el equilibrio en E=I=0.
        if self._learn_extra:
            ke = torch.sigmoid(-ae * thetae)
            ki = torch.sigmoid(-ai * thetai)
        else:
            ke, ki = self.ke, self.ki

        # MISMA estructura que rhs() en model.py (umbral adentro, offset ke/ki).
        u_i = wIE * E - wII * I + Q - thetai
        u_e = wEE * E - wEI * I + P - thetae

        # Correccion estructurada: factor refractario (1 - r*x) sobre la
        # sigmoidea. Con r=0 se recupera exactamente la forma reducida, y en el
        # reposo E=I=0 el factor vale 1, asi que el equilibrio se conserva.
        g_i = g_e = 1.0
        if self.structured:
            g_i = 1.0 - torch.clamp(self.raw_r_i, 0.0, 2.0) * I
            g_e = 1.0 - torch.clamp(self.raw_r_e, 0.0, 2.0) * E

        dI = (1.0 / ti) * (-I + g_i * torch.sigmoid(ai * u_i) - ki)
        dE = (1.0 / te) * (-E + g_e * torch.sigmoid(ae * u_e) - ke)
        return torch.cat([dI, dE], dim=-1)

    # Los parametros de la correccion estructurada, con su significado fisico.
    def structured_dict(self) -> dict[str, float]:
        if not self.structured:
            return {}
        alpha = float(torch.clamp(self.raw_alpha, 1e-3, 10.0).detach())
        return {"r_i": float(torch.clamp(self.raw_r_i, 0.0, 2.0).detach()),
                "r_e": float(torch.clamp(self.raw_r_e, 0.0, 2.0).detach()),
                "sat": 1.0 / alpha, "alpha": alpha}

    # -------------------------------------------------------------------------
    #  La correccion neuronal sola. Devuelve ceros si esta apagada.
    # -------------------------------------------------------------------------
    def g_out(self, x: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        if not self.use_correction:
            return torch.zeros_like(x[..., :2])
        I = x[..., 0:1]
        E = x[..., 1:2]
        if self.correction_inputs == "x":
            inp = torch.cat([I, E], dim=-1)
        else:
            P = torch.as_tensor(P, dtype=x.dtype, device=x.device) * torch.ones_like(I)
            Q = torch.as_tensor(Q, dtype=x.dtype, device=x.device) * torch.ones_like(I)
            inp = torch.cat([I, E, P, Q], dim=-1)
        return self.g(inp)

    # -------------------------------------------------------------------------
    #  ẋ = f_θ(x, P, Q) = backbone Wilson-Cowan + correccion neuronal.
    #  x: (...,2)=[I,E];  P,Q: tensores broadcastables a (...,1).
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, P, Q)
        if self.use_correction:
            out = out + self.g_out(x, P, Q)
        return out

    # -------------------------------------------------------------------------
    #  SENSIBILIDADES ∂f_backbone/∂θ  (10 parametros), por diferencias centradas.
    #
    #  Para que sirven: son las direcciones en las que un CAMBIO DE PARAMETROS
    #  puede mover el campo vectorial. Si la correccion neuronal se mueve en esas
    #  mismas direcciones, esta compitiendo con los parametros en vez de aportar
    #  fisica nueva -> es la base de la variante D (proyeccion ortogonal) y del
    #  diagnostico de identificabilidad del hibrido.
    #
    #  Devuelve (N, 2, 10), desconectado del grafo (es una referencia geometrica,
    #  no algo que haya que derivar).
    # -------------------------------------------------------------------------
    PARAM_ORDER = ("wEE", "wEI", "wIE", "wII", "te", "ti", "ae", "ai", "thetae", "thetai")

    @torch.no_grad()
    def backbone_sensitivities(self, x: torch.Tensor, P: torch.Tensor,
                               Q: torch.Tensor, rel_h: float = 1e-3) -> torch.Tensor:
        cols = []
        for name in self.PARAM_ORDER:
            if name in self.WEIGHTS:
                if not self._learn:
                    cols.append(torch.zeros_like(x[..., :2])); continue
                idx = self.WEIGHTS.index(name)
                raw, sel = self.raw_w, idx
            else:
                if not self._learn_extra:
                    cols.append(torch.zeros_like(x[..., :2])); continue
                raw, sel = getattr(self, f"raw_{name}"), None

            orig = raw.detach().clone()
            # paso relativo al valor actual del parametro (escala-invariante)
            base = orig[sel] if sel is not None else orig
            h = rel_h * (abs(float(base)) + 1.0)

            if sel is not None:
                raw.data[sel] = orig[sel] + h
            else:
                raw.data = orig + h
            f_plus = self.backbone(x, P, Q)

            if sel is not None:
                raw.data[sel] = orig[sel] - h
            else:
                raw.data = orig - h
            f_minus = self.backbone(x, P, Q)

            raw.data = orig.clone()
            cols.append((f_plus - f_minus) / (2.0 * h))
        return torch.stack(cols, dim=-1)
