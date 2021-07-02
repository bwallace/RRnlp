# RRnlp

This library provides (easy!) access to a suite of models for extracting key data from abstracts of randomized controlled trials. This is intended to provide access to lightweight versions of the models in Trialstreamer (https://trialstreamer.robotreviewer.net/; https://academic.oup.com/jamia/article/27/12/1903/5907063). However, the models here — all save for the sample size extractor constructed as linear layers on top of SciBERT representations, with only minimal fine tuning of SciBERT layers — are still experimental, and may not be as performant as the models used in Trialstreamer (yet!). 

# Use

```
    import rrnlp
    trial_reader = rrnlp.TrialReader()
    
    abstract = '''Background: Current strategies for preventing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are limited to nonpharmacologic interventions. Hydroxychloroquine has been proposed as a postexposure therapy to prevent coronavirus disease 2019 (Covid-19), but definitive evidence is lacking.\n\nMethods: We conducted an open-label, cluster-randomized trial involving asymptomatic contacts of patients with polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia, Spain. We randomly assigned clusters of contacts to the hydroxychloroquine group (which received the drug at a dose of 800 mg once, followed by 400 mg daily for 6 days) or to the usual-care group (which received no specific therapy). The primary outcome was PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary outcome was SARS-CoV-2 infection, defined by symptoms compatible with Covid-19 or a positive PCR test regardless of symptoms. Adverse events were assessed for up to 28 days.\n\nResults: The analysis included 2314 healthy contacts of 672 index case patients with Covid-19 who were identified between March 17 and April 28, 2020. A total of 1116 contacts were randomly assigned to receive hydroxychloroquine and 1198 to receive usual care. Results were similar in the hydroxychloroquine and usual-care groups with respect to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52 to 1.42]). In addition, hydroxychloroquine was not associated with a lower incidence of SARS-CoV-2 transmission than usual care (18.7% and 17.8%, respectively). The incidence of adverse events was higher in the hydroxychloroquine group than in the usual-care group (56.1% vs. 5.9%), but no treatment-related serious adverse events were reported.\n\nConclusions: Postexposure therapy with hydroxychloroquine did not prevent SARS-CoV-2 infection or symptomatic Covid-19 in healthy persons exposed to a PCR-positive case patient. (Funded by the crowdfunding campaign YoMeCorono and others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).'''
    
    preds = trial_reader.read_trial(abstract)
    print(preds)
```

Should yield the following dictionary

```
{   'PICO': {   'i': [   'Hydroxychloroquine',
                         'hydroxychloroquine',
                         'received',
                         'drug',
                         '800',
                         '400',
                         'usual-care',
                         'received no',
                         'therapy).',
                         'hydroxychloroquine',
                         'usual care.',
                         'hydroxychloroquine',
                         'usual-care',
                         'hydroxychloroquine',
                         'usual',
                         'hydroxychloroquine',
                         'usual-care',
                         'hydroxychloroquine'],
                'o': [   'symptomatic',
                         'SARS-CoV-2 infection,',
                         'symptoms',
                         'PCR test',
                         'Adverse events',
                         'incidence of PCR-confirmed, symptomatic Covid-19',
                         'incidence of SARS-CoV-2 transmission',
                         'incidence of adverse events',
                         'serious adverse events',
                         'infection'],
                'p': [   'acute respiratory syndrome coronavirus 2',
                         'disease',
                         'asymptomatic contacts of patients with '
                         'polymerase-chain-reaction',
                         'Covid-19 in Catalonia, Spain.',
                         'clusters of contacts',
                         '2314 healthy contacts of 672 index case patients '
                         'with Covid-19 who were identified between March 17 '
                         'and April',
                         '2020.',
                         'healthy persons exposed to a PCR-positive',
                         'patient.']},
    'ev_inf': (   'Results were similar in the hydroxychloroquine and '
                  'usual-care groups with respect to the incidence of '
                  'PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, '
                  'respectively; risk ratio, 0.86 [95% confidence interval, '
                  '0.52 to 1.42]).',
                  '↓ sig. decrease'),
    'n': '2314',
    'p_low_RoB': 0.000204605}
```

# Installing

Pull the repository and `cd` into `rrnlp`. 

Now, assuming conda is installed, one can proceed as follows

```
    conda create --name rrnlp
    conda activate rrnlp
    conda install pip
    pip install .
```

Alternatively, without pulling (but assuming `pip` available)

```
    python -m pip install https://github.com/bwallace/RRnlp/archive/refs/tags/alpha.tar.gz
```
