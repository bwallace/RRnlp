# RRnlp

This library provides (easy!) access to a suite of models for extracting key data from abstracts of randomized controlled trials (RCTs). 

In particular, `rrnlp` features lightweight variants of the models defined in Trialstreamer (https://trialstreamer.robotreviewer.net/; https://academic.oup.com/jamia/article/27/12/1903/5907063). However, the models here — all save for the sample size extractor constructed as linear layers on top of `SciBERT` representations, with only minimal fine tuning of `SciBERT` layers — are still experimental, and may not be as performant as the models used in Trialstreamer (yet!). See below for example usage. 

# Use

```python
import rrnlp

trial_reader = rrnlp.TrialReader()

ti_abs = {"ti": 'A Cluster-Randomized Trial of Hydroxychloroquine for Prevention of Covid-19',
          "ab": '''Background: Current strategies for preventing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are limited to nonpharmacologic interventions. Hydroxychloroquine has been proposed as a postexposure therapy to prevent coronavirus disease 2019 (Covid-19), but definitive evidence is lacking.\n\nMethods: We conducted an open-label, cluster-randomized trial involving asymptomatic contacts of patients with polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia, Spain. We randomly assigned clusters of contacts to the hydroxychloroquine group (which received the drug at a dose of 800 mg once, followed by 400 mg daily for 6 days) or to the usual-care group (which received no specific therapy). The primary outcome was PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary outcome was SARS-CoV-2 infection, defined by symptoms compatible with Covid-19 or a positive PCR test regardless of symptoms. Adverse events were assessed for up to 28 days.\n\nResults: The analysis included 2314 healthy contacts of 672 index case patients with Covid-19 who were identified between March 17 and April 28, 2020. A total of 1116 contacts were randomly assigned to receive hydroxychloroquine and 1198 to receive usual care. Results were similar in the hydroxychloroquine and usual-care groups with respect to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52 to 1.42]). In addition, hydroxychloroquine was not associated with a lower incidence of SARS-CoV-2 transmission than usual care (18.7% and 17.8%, respectively). The incidence of adverse events was higher in the hydroxychloroquine group than in the usual-care group (56.1% vs. 5.9%), but no treatment-related serious adverse events were reported.\n\nConclusions: Postexposure therapy with hydroxychloroquine did not prevent SARS-CoV-2 infection or symptomatic Covid-19 in healthy persons exposed to a PCR-positive case patient. (Funded by the crowdfunding campaign YoMeCorono and others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).'''}

preds = trial_reader.read_trial(ti_abs)
```

Should yield the following dictionary

```python
import pprint
pp = pprint.PrettyPrinter(width=200)
pp.pprint(preds)

{'bias_ab_bot': {'prob_low_rob': 0.14128409107623344},
'pico_span_bot': {'i': ['hydroxychloroquine', 'Hydroxychloroquine', 'usual care', 'drug', 'usual-care group (which received no specific therapy', 'hydroxychloroquine group'],
               'i_mesh': [{'cui': 'C0020336', 'mesh_term': 'Hydroxychloroquine', 'mesh_ui': 'D006886'},
                          {'cui': 'C0013227', 'mesh_term': 'Pharmaceutical Preparations', 'mesh_ui': 'D004364'},
                          {'cui': 'C1257890', 'mesh_term': 'Population Groups', 'mesh_ui': 'D044382'},
                          {'cui': 'C0087111', 'mesh_term': 'Therapeutics', 'mesh_ui': 'D013812'}],
               'o': ['PCR-confirmed, symptomatic Covid-19',
                     'SARS-CoV-2 infection',
                     'incidence of adverse events',
                     'symptomatic Covid-19',
                     'Adverse',
                     'serious adverse events',
                     'Covid-19 or a positive PCR test',
                     'SARS-CoV-2',
                     'incidence of PCR-confirmed, symptomatic Covid-19',
                     'incidence of SARS-CoV-2 transmission',
                     'symptoms'],
               'o_mesh': [{'cui': 'C0032520', 'mesh_term': 'Polymerase Chain Reaction', 'mesh_ui': 'D016133'},
                          {'cui': 'TS-COV19', 'mesh_term': 'COVID-19', 'mesh_ui': 'C000657245'},
                          {'cui': 'C1175743', 'mesh_term': 'SARS Virus', 'mesh_ui': 'D045473'},
                          {'cui': 'C3714514', 'mesh_term': 'Infection', 'mesh_ui': 'D007239'},
                          {'cui': 'C0021149', 'mesh_term': 'Incidence', 'mesh_ui': 'D015994'},
                          {'cui': 'C0040722', 'mesh_term': 'transmission', 'mesh_ui': 'Q000635'},
                          {'cui': 'C0683368', 'mesh_term': 'symptoms', 'mesh_ui': 'Q000175'}],
               'p': ['2314 healthy contacts of 672 index case patients with Covid-19 who were identified between March 17 and April 28, 2020',
                     'asymptomatic contacts of patients with polymerase-chain-reaction',
                     'healthy persons',
                     'Covid-19 in Catalonia, Spain',
                     'PCR-positive'],
               'p_mesh': [{'cui': 'C0600653', 'mesh_term': 'Index', 'mesh_ui': 'D020481'},
                          {'cui': 'C0030705', 'mesh_term': 'Patient', 'mesh_ui': 'D010361'},
                          {'cui': 'TS-COV19', 'mesh_term': 'COVID-19', 'mesh_ui': 'C000657245'},
                          {'cui': 'C0032520', 'mesh_term': 'Polymerase Chain Reaction', 'mesh_ui': 'D016133'},
                          {'cui': 'C0027361', 'mesh_term': 'Person', 'mesh_ui': 'D009272'},
                          {'cui': 'C0037747', 'mesh_term': 'Spain', 'mesh_ui': 'D013030'}]},
'punchline_bot': {'effect': '— no diff',
               'punchline_text': 'Results were similar in the hydroxychloroquine and usual-care groups with respect to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, '
                                 'respectively; risk ratio, 0.86 [95% confidence interval, 0.52 to 1.42]).'},
'rct_bot': {'is_rct': True, 'prob_rct': 0.6828127889603965, 'scores': {'is_rct_balanced': True, 'is_rct_precise': True, 'is_rct_sensitive': True}},
'sample_size_bot': {'num_randomized': '2314'}}
```

# Installing

The easiest way to install the latest version is via `pip`. 

```bash
pip install rrnlp
```

(Model weights will then be downloaded as needed when you import `rrnlp`.) We suggest using a custom environment, so if you're using `conda` this might look something like

```bash
conda create --name rrnlp python
conda activate rrnlp
pip install rrnlp
```

Alternatively, if you want to use the bleeding-edge (for better or worse) you can try installing directly via `git`

```bash
pip install git+https://github.com/bwallace/RRnlp.git
```

(Or can `clone` and then `install .` locally.)


# Citation 

This set of models is a compilation of several different lines of work. If you use this and find it useful for your work, please consider citing (some subset of) the following.

For the overall system: 

```
Marshall, I.J., Nye, B., Kuiper, J., Noel-Storr, A., Marshall, R., Maclean, R., Soboczenski, F., Nenkova, A., Thomas, J. and Wallace, B.C., 2020. Trialstreamer: A living, automatically updated database of clinical trial reports. Journal of the American Medical Informatics Association, 27(12), pp.1903-1912.

Nye, B.E., Nenkova, A., Marshall, I.J. and Wallace, B.C., 2020, July. Trialstreamer: mapping and browsing medical evidence in real-time. In Proceedings of the conference. Association for Computational Linguistics. North American Chapter. Meeting (Vol. 2020, p. 63). 
```

For the "inference" component specifically ("punchlines" and directionality):

```
Eric Lehman, Jay DeYoung, Regina Barzilay, and Byron C. Wallace. Inferring Which Medical Treatments Work from Reports of Clinical Trials. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), pages 3705–3717, 2019.

Jay DeYoung, Eric Lehman, Benjamin Nye, Iain Marshall, and Byron C. Wallace. Evidence Inference 2.0: More Data, Better Models. In Proceedings of BioNLP; co-located with the Association for Computational Linguistics (ACL), 2020.
```

If you are using the PICO snippets

```
Benjamin Nye, Jessy Li, Roma Patel, Yinfei Yang, Iain Marshall, Ani Nenkova, and Byron C. Wallace. A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature. In Proceedings of the Conference of the Association for Computational Linguistics (ACL), pages 197–207, 2018.
```

For the RCT classifier

```
Marshall, Iain J., Anna Noel‐Storr, Joël Kuiper, James Thomas, and Byron C. Wallace. "Machine learning for identifying randomized controlled trials: an evaluation and practitioner's guide." Research Synthesis Methods 9, no. 4 (2018): 602-614.
```

And for risk of bias

```
Iain J. Marshall, Joël Kuiper, and Byron C. Wallace. RobotReviewer: Evaluation of a System for Automatically Assessing Bias in Clinical Trials. Journal of the American Medical Informatics Association (JAMIA), 23(1):193–201, 2016.
```

# Support

This work has been supported by National Institutes of Health (NIH) under the National Library of Medicine (NLM), grant R01-LM012086 and by the National Science Foundation (NSF) under Grant 1750978: "CAREER: Structured Scientific Evidence Extraction: Models and Corpora". The work has also been partially supported by the UK Medical Research Council (MRC), through its Skills Development Fellowship program, fellowship MR/N015185/1.

