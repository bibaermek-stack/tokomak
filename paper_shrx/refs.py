"""Reference list (the order of the keys is irrelevant: the build numbers them by first citation).

Checked on 19.10.2026 against bibliographic records found on the web (publisher, repository and library pages): authors, title, journal,
volume, pages and year of every entry.  The publishers' own pages were not reachable from the build machine, so the page numbers should be
compared with the DOI pages once more before submission.  Zanisi et al. follows the CITATION.cff of the UKAEA repository itself.
"""

REFS = {
    "qlknn": "van de Plassche K. L., Citrin J., Bourdelle C., et al. Fast modeling of turbulent transport in fusion plasmas using neural networks // "
             "Physics of Plasmas. — 2020. — Vol. 27, No. 2. — P. 022310. — arXiv:1911.05617.",
    "torax": "Citrin J., Goodfellow I., Raju A., et al. TORAX: a fast and differentiable tokamak transport simulator "
             "in JAX // arXiv:2406.06718. — 2024.",
    "qualikiz": "Bourdelle C., Citrin J., Baiocchi B., et al. "
                "Core turbulent transport in tokamak plasmas: bridging theory and experiment with QuaLiKiz // "
                "Plasma Physics and Controlled Fusion. — 2016. — Vol. 58, No. 1. — P. 014036. — DOI: 10.1088/0741-3335/58/1/014036.",
    "tglf": "Staebler G. M., Kinsey J. E., Waltz R. E. A theory-based transport model with comprehensive physics // "
            "Physics of Plasmas. — 2007. — Vol. 14. — P. 055909.",
    "ipb": "ITER Physics Expert Group on Confinement and Transport, ITER Physics Expert Group on Confinement Modelling and Database, "
           "ITER Physics Basis Editors. ITER Physics Basis. Chapter 2: Plasma confinement and transport // Nuclear Fusion. — 1999. — "
           "Vol. 39. — P. 2175.",
    "connor": "Connor J. W., Taylor J. B. Scaling laws for plasma confinement // Nuclear Fusion. — 1977. — Vol. 17. — P. 1047.",
    "wesson": "Wesson J. Tokamaks. — 4th ed. — Oxford: Oxford University Press, 2011. — (International Series of Monographs on Physics; 149). — "
              "ISBN 978-0-19-959223-4.",
    "sauter": "Sauter O., Angioni C., Lin-Liu Y. R. Neoclassical conductivity and bootstrap current formulas for general axisymmetric "
              "equilibria and arbitrary collisionality regime // Physics of Plasmas. — 1999. — Vol. 6. — P. 2834. — DOI: 10.1063/1.873240; "
              "erratum: Physics of Plasmas. — 2002. — Vol. 9. — P. 5140.",
    "eped": "Snyder P. B. et al. A first-principles predictive model of the pedestal height and width: development, testing and ITER "
            "optimization with the EPED model // Nuclear Fusion. — 2011. — Vol. 51. — P. 103016.",
    "hinton": "Hinton G., Vinyals O., Dean J. Distilling the knowledge in a neural network // arXiv:1503.02531. — 2015 "
              "(NIPS 2014 Deep Learning Workshop).",
    "dagger": "Ross S., Gordon G., Bagnell D. A reduction of imitation learning and structured prediction to no-regret online learning // "
              "Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 15. — 2011. — "
              "P. 627–635.",
    "ensembles": "Lakshminarayanan B., Pritzel A., Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles // "
                 "Advances in Neural Information Processing Systems 30 (NeurIPS 2017). — 2017. — P. 6402–6413.",
    "adam": "Kingma D. P., Ba J. L. Adam: a method for stochastic optimization // International Conference on Learning Representations "
            "(ICLR 2015). — arXiv:1412.6980.",
    "he": "He K., Zhang X., Ren S., Sun J. Delving deep into rectifiers: surpassing human-level performance on ImageNet classification // "
          "Proceedings of the IEEE International Conference on Computer Vision (ICCV). — 2015. — P. 1026–1034. — DOI: 10.1109/ICCV.2015.123.",
    "roc": "Fawcett T. An introduction to ROC analysis // Pattern Recognition Letters. — 2006. — Vol. 27, No. 8. — P. 861–874. — "
           "DOI: 10.1016/j.patrec.2005.10.010.",
    "lorenz": "Lorenz E. N. Deterministic nonperiodic flow // Journal of the Atmospheric Sciences. — 1963. — Vol. 20. — P. 130–141.",
    "meneghini": "Meneghini O. et al. Self-consistent core-pedestal transport simulations with neural network accelerated models // "
                 "Nuclear Fusion. — 2017. — Vol. 57. — P. 086034. — DOI: 10.1088/1741-4326/aa7776.",
    "zanisi": "Zanisi L., Casson F., Snodin A., et al. Data-efficient digital twinning strategies and surrogate models of quasilinear "
              "turbulence in JET and STEP // IAEA Fusion Energy Conference (FEC 2025), 13–18 October 2025, Chengdu, China. — "
              "URL: https://conferences.iaea.org/event/392/contributions/36059/.",
    "tglfnn": "UKAEA. tglfnn-ukaea: neural network surrogates of TGLF [Электрондық ресурс]. — URL: https://github.com/ukaea/tglfnn-ukaea "
              "(қаралған күні: 19.10.2026).",
    "jax": "Bradbury J., Frostig R., Hawkins P., et al. JAX: composable transformations of Python+NumPy programs [Электрондық ресурс]. — "
           "2018. — URL: https://github.com/jax-ml/jax (қаралған күні: 19.10.2026).",
    "numpy": "Harris C. R., Millman K. J., van der Walt S. J., et al. Array programming with NumPy // Nature. — 2020. — Vol. 585. — "
             "P. 357–362. — DOI: 10.1038/s41586-020-2649-2.",
    "hdb5": "Verdoolaege G., Kaye S. M., Angioni C., et al. The updated ITPA global H-mode confinement database: description and analysis // "
            "Nuclear Fusion. — 2021. — Vol. 61. — P. 076006. Осы жұмыста қолданылған ашық үлгі FUSE.jl-мен бірге таратылады "
            "[Электрондық ресурс]. — URL: https://github.com/ProjectTorreyPines/FUSE.jl (файл sample/HDB5_compressed.csv, "
            "қаралған күні: 19.10.2026); дерекқор DOI 10.11578/1814950 бойынша да архивтелген.",
}
