# Alt-termination analysis

Panel: 30 bodies x 3 modes x 3 seeds = 270/270 valid runs.

## ANOVA (psr ~ C(mode) * C(family) + C(body))

- interaction F(4, 234) = 5.975, p = 0.000136

## Rank correlation with height-based Delta_reset

- Spearman rho(contact-Delta, height-Delta) = 0.188 [95% CI -0.180, 0.502] (n=30)

## Direction checks

- Floor strict - episodic: mean Delta = 0.836 [0.112, 1.489] (passes = False)
- VT strict - episodic: mean Delta = -0.220 [-0.814, 0.342] (passes = False)

## Gate decision

**drop**

Source: interaction_p=0.000136, rho=0.188, floor-dir=False, vt-dir=False.
