import rrnlp

meta_reviewer = rrnlp.MetaReviewer()

abs = ['Ninety-four patients with infectious mononucleosis and symptoms < or = 7 days were randomized to treatment with oral acyclovir (800 mg 5 times/day) and prednisolone (0.7 mg/kg for the first 4 days, which was reduced by 0.1 mg/kg on consecutive days for another 6 days; n = 48), or placebo (n = 46) for 10 days. Oropharyngeal Epstein-Barr virus (EBV) shedding was significantly inhibited during the treatment period (P = .02, Mann-Whitney rank test). No significant effect was observed for duration of general illness, sore throat, weight loss, or absence from school or work. The frequency of latent EBV-infected B lymphocytes in peripheral blood and the HLA-restricted EBV-specific cellular immunity, measured 6 months after onset of disease, was not affected by treatment. Thus, acyclovir combined with prednisolone inhibited oropharyngeal EBV replication without affecting duration of clinical symptoms or development of EBV-specific cellular immunity.']

print(meta_reviewer.summarize(abs))
