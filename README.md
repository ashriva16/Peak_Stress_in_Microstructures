<p align="left">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8%2B-blue" />
  <img alt="Repo Size" src="https://img.shields.io/github/repo-size/ashriva16/Peak_Stress_in_Microstructures" />
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/ashriva16/Peak_Stress_in_Microstructures" />
  <a href="https://github.com/ashriva16/Peak_Stress_in_Microstructures/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/ashriva16/Peak_Stress_in_Microstructures" />
  </a>
  <a href="https://github.com/ashriva16/Peak_Stress_in_Microstructures/pulls">
    <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/ashriva16/Peak_Stress_in_Microstructures" />
  </a>
</p>

# Deep Learning for Peak-Stress Prediction in Polycrystalline Materials

## 📌 Project Description
This project presents a deep learning approach to predict peak-stress clusters in heterogeneous polycrystalline materials. Unlike prior work that focused on overall stress fields, this method targets localized peak-stress regions critical to failure. Using a convolutional encoder–decoder network trained on synthetic microstructures and linear elasticity simulations, the model predicts stress fields and identifies peak-stress clusters. Evaluation using cosine similarity and geometric comparisons shows high accuracy, especially for higher normalized stress values.

![Highlight](highlight.png)

### ✅ Key Features

- Rapid identification of optimal deposition parameters
- Improved consistency and reproducibility of thin film properties
- Reduced experimental effort

Our results confirm that Bayesian optimization is a powerful tool for thin film process development, delivering high-performance films with controlled stress and resistance characteristics.

---

## 🧱 Project Structure

```text
.
├── LICENSE             # Licensing information (e.g., MIT, Apache 2.0)
├── README.md           # Project overview, usage, setup, and contribution guidelines
├── requirement.txt
├── train.py           # main file
├── data_loader/        # Scripts for loading and preprocessing data
├── models/             # Model architecture definitions and related code
├── trainer/            # Training loops and experiment execution logic
├── utils/              # Shared utility functions and helper modules
```

---

Badge (once setup):

```markdown
[![CI](https://github.com/ashriva16/Peak_Stress_in_Microstructures/actions/workflows/ci.yml/badge.svg)](https://github.com/ashriva16/Peak_Stress_in_Microstructures/actions)
```

---

## 👤 Maintainer

**Ankit Shrivastava**
Feel free to open an issue or discussion for support.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for full details.

---

## 📈 Project Status

> Status: ✅ Ready for Use — Not Actively Maintained
---

## 📘 References

- [Cookiecutter Docs](https://cookiecutter.readthedocs.io)
- [PEP 621](https://peps.python.org/pep-0621/)
- [GitHub Actions](https://docs.github.com/en/actions)
