# Comparación de controladores: tablas

Mediana del RMSE de seguimiento sobre niveles de ruido y semillas. «solo β̂» = controlador original con los parámetros del modelo; «con modelo» = original cancelando g o con la forma exacta; «extendido» = despeje con la red o con la copia del actuador. Tope: P en [0, 2,0], Q en [0, 1,5].

## modelos de 2 s, referencia gamma 50 ms

**actuador**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0692 | 0.3975 | 0.0730 | [-13.4, 12.9] | 84 % |
| white-box | — | original, solo β̂ | 0.0897 | 0.3972 | 0.0646 | [-15.2, 9.7] | 75 % |
| agnóstica | desde cero | original, con modelo | 0.1359 | 0.3771 | 0.0901 | [-13.1, 12.4] | 74 % |
| agnóstica | desde cero | original, solo β̂ | 0.0802 | 0.3972 | 0.0634 | [-13.5, 7.3] | 75 % |
| agnóstica | warm start | original, con modelo | 0.1278 | 0.3940 | 0.0654 | [-15.3, 14.3] | 75 % |
| agnóstica | warm start | original, solo β̂ | 0.0834 | 0.3973 | 0.0659 | [-11.7, 11.8] | 80 % |
| historia | desde cero | extendido | 0.0698 | 0.3978 | 0.0743 | [-12.8, 11.9] | 94 % |
| historia | desde cero | original, solo β̂ | 0.0847 | 0.3970 | 0.0669 | [-12.9, 6.6] | 72 % |
| historia | warm start | extendido | 0.0551 | 0.3977 | 0.0766 | [-18.6, 16.9] | 94 % |
| historia | warm start | original, solo β̂ | 0.0846 | 0.3971 | 0.0638 | [-8.2, 16.8] | 77 % |
| filtro | desde cero | extendido | 0.0834 | 0.3946 | 0.0872 | [-14.4, 13.2] | 91 % |
| filtro | desde cero | original, solo β̂ | 0.0818 | 0.3967 | 0.0667 | [-13.7, 7.3] | 74 % |
| filtro | warm start | extendido | 0.0775 | 0.3954 | 0.0650 | [-18.9, 17.1] | 89 % |
| filtro | warm start | original, solo β̂ | 0.0737 | 0.3969 | 0.0647 | [-9.2, 16.9] | 77 % |
| estado de filtro | desde cero | extendido | 0.0328 | 0.3959 | 0.0410 | [-28.2, 72.2] | 62 % |
| estado de filtro | desde cero | original, solo β̂ | 0.0766 | 0.3975 | 0.0678 | [-6.6, 14.1] | 81 % |
| estado de filtro | warm start | extendido | 0.0332 | 0.3960 | 0.0410 | [-28.6, 73.2] | 61 % |
| estado de filtro | warm start | original, solo β̂ | 0.0766 | 0.3975 | 0.0668 | [-6.7, 15.0] | 80 % |

**refractariedad**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0339 | 0.0422 | 0.0364 | [0.8, 12.4] | 43 % |
| oráculo + forma | — | oráculo + forma | 0.0324 | 0.0408 | 0.0351 | [0.8, 12.4] | 43 % |
| white-box | — | original, solo β̂ | 0.0336 | 0.0418 | 0.0361 | [0.8, 8.9] | 43 % |
| agnóstica 5 ms | desde cero | original, con modelo | 0.0453 | 0.0449 | 0.0421 | [0.8, 10.5] | 33 % |
| agnóstica 5 ms | desde cero | original, solo β̂ | 0.0348 | 0.0426 | 0.0370 | [0.8, 6.4] | 42 % |
| agnóstica 5 ms | warm start | original, con modelo | 0.0339 | 0.0412 | 0.0355 | [0.8, 11.8] | 43 % |
| agnóstica 5 ms | warm start | original, solo β̂ | 0.0341 | 0.0421 | 0.0364 | [0.8, 11.8] | 43 % |
| agnóstica 20 ms | desde cero | original, con modelo | 0.0444 | 0.0446 | 0.0415 | [0.8, 10.5] | 33 % |
| agnóstica 20 ms | desde cero | original, solo β̂ | 0.0344 | 0.0424 | 0.0368 | [0.8, 6.4] | 43 % |
| agnóstica 20 ms | warm start | original, con modelo | 0.0340 | 0.0413 | 0.0355 | [0.8, 11.9] | 43 % |
| agnóstica 20 ms | warm start | original, solo β̂ | 0.0341 | 0.0421 | 0.0364 | [0.8, 11.9] | 43 % |
| forma exacta | warm start | original, con modelo | 0.0326 | 0.0408 | 0.0351 | [0.8, 12.2] | 43 % |
| forma exacta | warm start | original, solo β̂ | 0.0337 | 0.0419 | 0.0362 | [0.8, 9.7] | 43 % |

## modelos de 2 s, referencia theta-gamma 2 s

**actuador**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0599 | 0.2001 | 0.0824 | [-13.7, 13.1] | 66 % |
| white-box | — | original, solo β̂ | 0.0644 | 0.1994 | 0.0820 | [-15.5, 10.7] | 82 % |
| agnóstica | desde cero | original, con modelo | 0.0898 | 0.1873 | 0.0851 | [-14.3, 13.1] | 93 % |
| agnóstica | desde cero | original, solo β̂ | 0.0686 | 0.1995 | 0.0822 | [-14.3, 8.3] | 80 % |
| agnóstica | warm start | original, con modelo | 0.0802 | 0.1942 | 0.0821 | [-16.7, 15.4] | 83 % |
| agnóstica | warm start | original, solo β̂ | 0.0672 | 0.1995 | 0.0820 | [-11.9, 12.5] | 82 % |
| historia | desde cero | extendido | 0.0629 | 0.2012 | 0.0865 | [-13.4, 12.2] | 96 % |
| historia | desde cero | original, solo β̂ | 0.0681 | 0.1995 | 0.0822 | [-13.0, 7.1] | 65 % |
| historia | warm start | extendido | 0.0640 | 0.1980 | 0.0828 | [-19.4, 17.5] | 90 % |
| historia | warm start | original, solo β̂ | 0.0712 | 0.1994 | 0.0818 | [-8.6, 17.5] | 82 % |
| filtro | desde cero | extendido | 0.0664 | 0.2004 | 0.0858 | [-15.0, 13.4] | 96 % |
| filtro | desde cero | original, solo β̂ | 0.0637 | 0.1989 | 0.0821 | [-14.0, 7.6] | 77 % |
| filtro | warm start | extendido | 0.0659 | 0.1991 | 0.0822 | [-19.4, 17.3] | 96 % |
| filtro | warm start | original, solo β̂ | 0.0666 | 0.1991 | 0.0818 | [-9.3, 17.2] | 82 % |
| estado de filtro | desde cero | extendido | 0.0432 | 0.1935 | 0.0793 | [-36.7, 72.2] | 39 % |
| estado de filtro | desde cero | original, solo β̂ | 0.0711 | 0.2001 | 0.0825 | [-6.7, 14.5] | 73 % |
| estado de filtro | warm start | extendido | 0.0434 | 0.1936 | 0.0793 | [-35.3, 73.2] | 38 % |
| estado de filtro | warm start | original, solo β̂ | 0.0712 | 0.2000 | 0.0824 | [-6.9, 15.5] | 79 % |

**refractariedad**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0386 | 0.0593 | 0.0510 | [-12.9, 12.4] | 28 % |
| oráculo + forma | — | oráculo + forma | 0.0384 | 0.0593 | 0.0509 | [-12.8, 12.4] | 27 % |
| white-box | — | original, solo β̂ | 0.0387 | 0.0595 | 0.0510 | [-11.8, 8.9] | 28 % |
| agnóstica 5 ms | desde cero | original, con modelo | 0.0428 | 0.0604 | 0.0532 | [-8.6, 10.7] | 84 % |
| agnóstica 5 ms | desde cero | original, solo β̂ | 0.0389 | 0.0595 | 0.0512 | [-11.0, 6.4] | 33 % |
| agnóstica 5 ms | warm start | original, con modelo | 0.0395 | 0.0595 | 0.0510 | [-9.1, 11.8] | 29 % |
| agnóstica 5 ms | warm start | original, solo β̂ | 0.0387 | 0.0593 | 0.0510 | [-6.2, 11.8] | 29 % |
| agnóstica 20 ms | desde cero | original, con modelo | 0.0429 | 0.0605 | 0.0534 | [-8.6, 10.7] | 85 % |
| agnóstica 20 ms | desde cero | original, solo β̂ | 0.0388 | 0.0594 | 0.0511 | [-11.1, 6.4] | 33 % |
| agnóstica 20 ms | warm start | original, con modelo | 0.0395 | 0.0595 | 0.0510 | [-9.2, 11.9] | 30 % |
| agnóstica 20 ms | warm start | original, solo β̂ | 0.0387 | 0.0593 | 0.0510 | [-6.5, 11.9] | 31 % |
| forma exacta | warm start | original, con modelo | 0.0385 | 0.0594 | 0.0509 | [-12.2, 12.2] | 32 % |
| forma exacta | warm start | original, solo β̂ | 0.0386 | 0.0593 | 0.0510 | [-12.2, 9.7] | 31 % |

## modelos de 200 ms, referencia gamma 50 ms

**actuador**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0692 | 0.3975 | 0.0730 | [-13.4, 12.9] | 84 % |
| white-box | — | original, solo β̂ | 0.0894 | 0.3972 | 0.0625 | [-15.5, 9.0] | 75 % |
| agnóstica | desde cero | original, con modelo | 0.1607 | 0.3793 | 0.0657 | [-13.3, 12.7] | 60 % |
| agnóstica | desde cero | original, solo β̂ | 0.0939 | 0.3974 | 0.0644 | [-13.1, 6.8] | 76 % |
| agnóstica | warm start | original, con modelo | 0.1125 | 0.3894 | 0.0638 | [-15.5, 13.9] | 76 % |
| agnóstica | warm start | original, solo β̂ | 0.0798 | 0.3973 | 0.0639 | [-15.3, 8.5] | 80 % |
| historia | desde cero | extendido | 0.0935 | 0.3958 | 0.0673 | [-11.6, 11.0] | 80 % |
| historia | desde cero | original, solo β̂ | 0.0824 | 0.3973 | 0.0685 | [-11.2, 6.0] | 80 % |
| historia | warm start | extendido | 0.0644 | 0.3958 | 0.0682 | [-16.1, 15.0] | 87 % |
| historia | warm start | original, solo β̂ | 0.0878 | 0.3972 | 0.0630 | [-16.2, 9.7] | 76 % |
| filtro | desde cero | extendido | 0.1047 | 0.3925 | 0.0809 | [-10.4, 10.2] | 82 % |
| filtro | desde cero | original, solo β̂ | 0.0910 | 0.3952 | 0.0697 | [-10.1, 5.5] | 79 % |
| filtro | warm start | extendido | 0.0964 | 0.3915 | 0.0653 | [-15.2, 14.3] | 83 % |
| filtro | warm start | original, solo β̂ | 0.0857 | 0.3970 | 0.0622 | [-15.6, 8.1] | 76 % |
| estado de filtro | desde cero | extendido | 0.0338 | 0.3959 | 0.0417 | [-24.8, 65.9] | 59 % |
| estado de filtro | desde cero | original, solo β̂ | 0.0781 | 0.3975 | 0.0679 | [-6.6, 14.4] | 80 % |
| estado de filtro | warm start | extendido | 0.0337 | 0.3959 | 0.0416 | [-25.2, 66.7] | 60 % |
| estado de filtro | warm start | original, solo β̂ | 0.0769 | 0.3974 | 0.0672 | [-6.7, 14.5] | 80 % |

**refractariedad**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0339 | 0.0422 | 0.0364 | [0.8, 12.4] | 43 % |
| oráculo + forma | — | oráculo + forma | 0.0324 | 0.0408 | 0.0351 | [0.8, 12.4] | 43 % |
| white-box | — | original, solo β̂ | 0.0335 | 0.0419 | 0.0360 | [0.8, 9.2] | 43 % |
| agnóstica 5 ms | desde cero | original, con modelo | 0.0460 | 0.0454 | 0.0429 | [0.8, 10.8] | 34 % |
| agnóstica 5 ms | desde cero | original, solo β̂ | 0.0349 | 0.0427 | 0.0371 | [0.8, 6.7] | 42 % |
| agnóstica 5 ms | warm start | original, con modelo | 0.0344 | 0.0413 | 0.0356 | [0.8, 11.8] | 43 % |
| agnóstica 5 ms | warm start | original, solo β̂ | 0.0342 | 0.0423 | 0.0365 | [0.8, 11.6] | 43 % |
| agnóstica 20 ms | desde cero | original, con modelo | 0.0462 | 0.0456 | 0.0431 | [0.8, 10.6] | 33 % |
| agnóstica 20 ms | desde cero | original, solo β̂ | 0.0345 | 0.0425 | 0.0370 | [0.8, 6.4] | 42 % |
| agnóstica 20 ms | warm start | original, con modelo | 0.0339 | 0.0413 | 0.0356 | [0.8, 11.9] | 43 % |
| agnóstica 20 ms | warm start | original, solo β̂ | 0.0341 | 0.0422 | 0.0365 | [0.8, 11.9] | 43 % |
| forma exacta | desde cero | original, con modelo | 0.0335 | 0.0419 | 0.0360 | [0.8, 9.8] | 44 % |
| forma exacta | desde cero | original, solo β̂ | 0.0335 | 0.0419 | 0.0360 | [0.8, 9.8] | 44 % |
| forma exacta | warm start | original, con modelo | 0.0327 | 0.0410 | 0.0352 | [0.8, 12.2] | 43 % |
| forma exacta | warm start | original, solo β̂ | 0.0338 | 0.0421 | 0.0363 | [0.8, 8.8] | 43 % |

## modelos de 200 ms, referencia theta-gamma 2 s

**actuador**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0599 | 0.2001 | 0.0824 | [-13.7, 13.1] | 66 % |
| white-box | — | original, solo β̂ | 0.0662 | 0.1989 | 0.0819 | [-15.8, 10.2] | 83 % |
| agnóstica | desde cero | original, con modelo | 0.0924 | 0.1879 | 0.0846 | [-13.7, 13.2] | 91 % |
| agnóstica | desde cero | original, solo β̂ | 0.0694 | 0.1994 | 0.0821 | [-13.1, 7.8] | 81 % |
| agnóstica | warm start | original, con modelo | 0.0753 | 0.1918 | 0.0820 | [-16.3, 15.4] | 84 % |
| agnóstica | warm start | original, solo β̂ | 0.0651 | 0.1992 | 0.0819 | [-15.6, 9.6] | 83 % |
| historia | desde cero | extendido | 0.0715 | 0.2001 | 0.0836 | [-12.0, 11.7] | 93 % |
| historia | desde cero | original, solo β̂ | 0.0708 | 0.1996 | 0.0824 | [-11.8, 6.9] | 76 % |
| historia | warm start | extendido | 0.0626 | 0.1977 | 0.0834 | [-16.9, 16.0] | 95 % |
| historia | warm start | original, solo β̂ | 0.0649 | 0.1991 | 0.0818 | [-16.3, 10.8] | 82 % |
| filtro | desde cero | extendido | 0.0736 | 0.1997 | 0.0849 | [-11.1, 10.5] | 94 % |
| filtro | desde cero | original, solo β̂ | 0.0720 | 0.1960 | 0.0822 | [-10.7, 5.7] | 64 % |
| filtro | warm start | extendido | 0.0702 | 0.1939 | 0.0825 | [-16.1, 15.1] | 95 % |
| filtro | warm start | original, solo β̂ | 0.0666 | 0.1987 | 0.0818 | [-15.7, 9.3] | 82 % |
| estado de filtro | desde cero | extendido | 0.0437 | 0.1937 | 0.0792 | [-33.0, 65.9] | 38 % |
| estado de filtro | desde cero | original, solo β̂ | 0.0718 | 0.2000 | 0.0824 | [-7.1, 14.8] | 79 % |
| estado de filtro | warm start | extendido | 0.0437 | 0.1936 | 0.0792 | [-32.8, 66.7] | 38 % |
| estado de filtro | warm start | original, solo β̂ | 0.0719 | 0.2000 | 0.0824 | [-7.1, 15.0] | 80 % |

**refractariedad**

| modelo | arranque | controlador | sin tope | tope literal | tope y anti-windup | P sin tope | contra el tope (aw) |
|---|---|---|---|---|---|---|---|
| oráculo | — | oráculo | 0.0386 | 0.0593 | 0.0510 | [-12.9, 12.4] | 27 % |
| oráculo + forma | — | oráculo + forma | 0.0384 | 0.0593 | 0.0509 | [-12.8, 12.4] | 28 % |
| white-box | — | original, solo β̂ | 0.0387 | 0.0594 | 0.0510 | [-11.3, 9.2] | 28 % |
| agnóstica 5 ms | desde cero | original, con modelo | 0.0431 | 0.0606 | 0.0533 | [-8.7, 10.9] | 85 % |
| agnóstica 5 ms | desde cero | original, solo β̂ | 0.0389 | 0.0595 | 0.0512 | [-11.4, 6.7] | 33 % |
| agnóstica 5 ms | warm start | original, con modelo | 0.0395 | 0.0595 | 0.0510 | [-9.2, 11.8] | 13 % |
| agnóstica 5 ms | warm start | original, solo β̂ | 0.0387 | 0.0594 | 0.0510 | [-6.7, 11.6] | 31 % |
| agnóstica 20 ms | desde cero | original, con modelo | 0.0434 | 0.0606 | 0.0537 | [-8.5, 10.7] | 85 % |
| agnóstica 20 ms | desde cero | original, solo β̂ | 0.0388 | 0.0594 | 0.0511 | [-11.1, 6.4] | 33 % |
| agnóstica 20 ms | warm start | original, con modelo | 0.0394 | 0.0594 | 0.0510 | [-9.2, 11.9] | 13 % |
| agnóstica 20 ms | warm start | original, solo β̂ | 0.0387 | 0.0593 | 0.0510 | [-6.9, 11.9] | 33 % |
| forma exacta | desde cero | original, con modelo | 0.0387 | 0.0594 | 0.0510 | [-11.9, 9.8] | 28 % |
| forma exacta | desde cero | original, solo β̂ | 0.0387 | 0.0594 | 0.0510 | [-11.9, 9.8] | 28 % |
| forma exacta | warm start | original, con modelo | 0.0385 | 0.0594 | 0.0509 | [-12.7, 12.2] | 15 % |
| forma exacta | warm start | original, solo β̂ | 0.0386 | 0.0593 | 0.0510 | [-12.6, 8.8] | 22 % |

