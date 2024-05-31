# CSIML

**CSIML** is a "cost-sensitive and iterative machine-learning" method for small and imbalanced materials data sets.

## Cite it

**S Li**, A Nakata*. CSIML: a cost-sensitive and iterative machine-learning method for small and imbalanced materials data sets[J]. *Chemistry Letters*, 2024, 53(5). [[DOI]](https://doi.org/10.1093/chemle/upae090)

## Usage

### Requirements

Here are some necessary environments for minimal main features:

| Enviroment Name | Version |
| :---: | :---: |
| Python | >=3.10 |
| zhonger/matminer | >=0.7.8 |
| scikit-learn | >=1.1.1 |
| mendeleev | >=0.10.0 |
| pandas | >=1.4.3 |
| numpy | >=1.23.1 |
| tqdm | >=4.64.0 |

Besieds these, you can install others in `requirements.txt` for more features.

### Build

```shell
pip3 install poetry
poetry build
pip3 install dist/*.whl
```
