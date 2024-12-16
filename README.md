# Regional Expected Improvement for Efficient Trust Region Selection in High-Dimensional Bayesian Optimization

This repository provides the source code proposed in the following article. Please cite this article if you use the code.

Nobuo Namura and Sho Takemori, "Regional Expected Improvement for Efficient Trust Region Selection
in High-Dimensional Bayesian Optimization," In Proceedings of the 39th AAAI Conference on Artificial Intelligence (2025).

## Requirements
To install requirements:

```
pip install -r requirements.txt
```

## Usage
1. Install the libraries listed in `requirements.txt` (see above).
2. Edit the `main.py` for your setting.
3. Run `python main.py`.

* To use the HPA problems, you need to download or "git clone" [this repository](https://github.com/Nobuo-Namura/hpa), and place it into `src/benchmark/hpa`
    * Directory tree must be as follows  
        ```
        regional-expected-improvement
        └── src/
            └── benchmark/
                ├── ebo
                ├── hpa/
                │   ├── hpa/
                │   │   ├── airfoil_info
                │   │   ├── __init__.py
                │   │   ├── adapter.py
                │   │   ├── designer.py
                │   │   └── problem.py
                │   ├── igd_reference_points
                │   ├── img
                │   └── utopia_and_nadir_points
                └── mopta08
        ```
* To use the MOPTA08 problem, you need to download binaries from [here](https://github.com/LeoIV/BenchSuite/tree/master/data/mopta08), and place them into `src/benchmark/mopta08`


## License
* This project is under the MIT License. See [LICENSE](LICENSE) for details.  