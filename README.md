# Gastric Cancer Classification

This repository refers to an on-going work for a multiclass classification problem in gastric cancer histopathology.

## Dataset
**Name**: HMU-GC-HE-30K – *Gastric Cancer Histopathology Tissue Image Dataset (GCHTID)*  
**Source**: Lou *et al.* “A large histological images dataset of gastric cancer with tumour microenvironment annotation for AI”. DOI: 10.1038/s41597-025-04489-9.
Avaliable at https://www.nature.com/articles/s41597-025-04489-9.

**Content**: ~31,000 RGB image patches (224 × 224 px) cropped from 300 H&E-stained whole-slide images (WSI) of gastric cancer patients.

Each patch is annotated with one of the eight tumour-microenvironment (TME) tissue classes:

| Abbr. | Tissue / Component     |
|-------|------------------------|
| **ADI** | Adipose tissue        |
| **DEB** | Debris                |
| **MUC** | Mucus                 |
| **MUS** | Muscle                |
| **LYM** | Lymphocyte aggregates |
| **STR** | Stroma                |
| **NOR** | Normal mucosa         |
| **TUM** | Tumour epithelium     |

