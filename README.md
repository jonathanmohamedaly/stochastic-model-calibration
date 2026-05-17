# Stochastic Model Calibration for Derivatives Pricing

## Overview

This project implements and calibrates several classical quantitative finance models used for derivatives pricing and implied volatility modelling.

The objective is to build a robust calibration framework combining:

- stochastic volatility modelling,
- interest rate modelling,
- numerical pricing,
- implied volatility inversion,
- constrained optimization,
- synthetic validation,
- and real market data calibration.

The models implemented are:

- **Heston stochastic volatility model**
- **SABR stochastic volatility model**
- **Hull-White one-factor interest rate model**

This project was designed as a quantitative finance portfolio project focused on model implementation, calibration, numerical methods, and financial engineering best practices.

---

# Models Implemented

## 1. Black-Scholes Framework

Implemented as the pricing benchmark and implied volatility inversion engine.

### Features

- European call pricing
- European put pricing
- Vega computation
- Implied volatility inversion:
  - Newton-Raphson solver
  - Brent robust root-finding solver

### Validation

- known analytical pricing values
- put-call parity
- synthetic implied volatility recovery

---

## 2. SABR Model

Implementation of the stochastic alpha beta rho model using Hagan’s asymptotic approximation.

### Dynamics

\[
dF_t = \alpha_t F_t^\beta dW_t
\]

\[
d\alpha_t = \nu \alpha_t dZ_t
\]

with:

\[
corr(dW_t, dZ_t) = \rho
\]

### Features

- parameter validation
- Hagan implied volatility approximation
- ATM handling
- smile calibration
- synthetic smile generation
- real market smile calibration

### Calibration

SABR is calibrated independently for each maturity by minimizing implied volatility fitting error.

Calibrated parameters:

- alpha
- rho
- nu

with fixed beta.

### Validation

- synthetic parameter recovery
- smile fitting diagnostics
- comparison against real option smiles

---

## 3. Heston Model

Implementation of the Heston stochastic volatility framework.

### Dynamics

\[
dS_t = (r-q)S_tdt + \sqrt{v_t}S_tdW_t
\]

\[
dv_t = \kappa(\theta-v_t)dt + \sigma \sqrt{v_t}dZ_t
\]

with:

\[
corr(dW_t,dZ_t)=\rho
\]

### Features

- parameter validation
- Feller condition diagnostics
- characteristic function implementation
- Fourier pricing
- European call pricing
- European put pricing
- Black-Scholes implied volatility conversion

### Calibration

Global implied volatility surface calibration using:

- constrained optimization
- numerical pricing
- volatility surface fitting

Parameters calibrated:

- kappa
- theta
- sigma
- rho
- v0

### Validation

- characteristic function sanity checks
- put-call parity
- synthetic volatility surface calibration
- real market calibration using SPY options

---

## 4. Hull-White One-Factor Model

Implementation of an interest rate derivatives pricing framework.

### Dynamics

\[
dr_t = [\theta(t)-ar_t]dt + \sigma dW_t
\]

### Features

- discount factor construction
- zero-coupon bond pricing components
- bond option pricing
- forward swap rate computation
- swap annuity computation
- payer swaption pricing
- receiver swaption pricing

### Calibration

Synthetic swaption surface calibration:

Parameters calibrated:

- mean reversion
- short-rate volatility

### Validation

- zero-coupon pricing consistency
- swaption put-call parity
- synthetic swaption parameter recovery
- integration with real Treasury yield curve data

---

# Project Structure

```text
stochastic-model-calibration/
│
├── data/
│
├── notebooks/
│   ├── market_data.ipynb
│   ├── heston_calibration.ipynb
│   ├── sabr_calibration.ipynb
│   ├── hull_white_calibration.ipynb
│   ├── sabr_vs_heston.ipynb
│   └── real_curve_hull_white.ipynb
│
├── src/
│   ├── models/
│   ├── pricing/
│   ├── calibration/
│   ├── visualization/
│   └── utils/
│
├── tests/
│
├── report/
│
├── README.md
├── requirements.txt
└── pyproject.toml

```
----------------------------------------------

# Calibration de Modèles Stochastiques pour le Pricing de Produits Dérivés

## Vue d’ensemble

Ce projet implémente et calibre plusieurs modèles classiques de finance quantitative utilisés pour le pricing de produits dérivés et la modélisation de la volatilité implicite.

L’objectif est de construire un framework de calibration robuste combinant :

- la modélisation de volatilité stochastique,
- la modélisation des taux d’intérêt,
- le pricing numérique,
- l’inversion de volatilité implicite,
- l’optimisation sous contraintes,
- la validation sur données synthétiques,
- la calibration sur données de marché réelles.

Les modèles implémentés sont :

- **Modèle de volatilité stochastique de Heston**
- **Modèle de volatilité stochastique SABR**
- **Modèle de taux d’intérêt Hull-White à un facteur**

Ce projet a été conçu comme un projet de portfolio en finance quantitative, centré sur l’implémentation de modèles, la calibration, les méthodes numériques et les bonnes pratiques d’ingénierie financière.

---

# Modèles implémentés

## 1. Cadre Black-Scholes

Implémenté comme benchmark de pricing et moteur d’inversion de volatilité implicite.

### Fonctionnalités

- Pricing d’options call européennes
- Pricing d’options put européennes
- Calcul du Vega
- Inversion de volatilité implicite :
  - solveur Newton-Raphson
  - solveur robuste de recherche de racine de Brent

### Validation

- valeurs analytiques de pricing connues
- parité put-call
- récupération de volatilité implicite sur données synthétiques

---

## 2. Modèle SABR

Implémentation du modèle stochastique alpha-beta-rho utilisant l’approximation asymptotique de Hagan.

### Dynamique

\[
dF_t = \alpha_t F_t^\beta dW_t
\]

\[
d\alpha_t = \nu \alpha_t dZ_t
\]

avec :

\[
corr(dW_t, dZ_t) = \rho
\]

### Fonctionnalités

- validation des paramètres
- approximation de volatilité implicite de Hagan
- gestion du cas ATM
- calibration du smile
- génération de smiles synthétiques
- calibration sur smiles de marché réels

### Calibration

Le modèle SABR est calibré indépendamment pour chaque maturité en minimisant l’erreur d’ajustement sur la volatilité implicite.

Paramètres calibrés :

- alpha
- rho
- nu

avec beta fixé.

### Validation

- récupération de paramètres synthétiques
- diagnostics d’ajustement du smile
- comparaison avec des smiles d’options réelles

---

## 3. Modèle de Heston

Implémentation du cadre de volatilité stochastique de Heston.

### Dynamique

\[
dS_t = (r-q)S_tdt + \sqrt{v_t}S_tdW_t
\]

\[
dv_t = \kappa(\theta-v_t)dt + \sigma \sqrt{v_t}dZ_t
\]

avec :

\[
corr(dW_t,dZ_t)=\rho
\]

### Fonctionnalités

- validation des paramètres
- diagnostics de la condition de Feller
- implémentation de la fonction caractéristique
- pricing par transformée de Fourier
- pricing d’options call européennes
- pricing d’options put européennes
- conversion vers volatilité implicite Black-Scholes

### Calibration

Calibration globale de la surface de volatilité implicite via :

- optimisation sous contraintes
- pricing numérique
- ajustement de surface de volatilité

Paramètres calibrés :

- kappa
- theta
- sigma
- rho
- v0

### Validation

- vérifications de cohérence de la fonction caractéristique
- parité put-call
- calibration sur surface de volatilité synthétique
- calibration réelle sur options SPY

---

## 4. Modèle Hull-White à un facteur

Implémentation d’un framework de pricing pour produits dérivés de taux.

### Dynamique

\[
dr_t = [\theta(t)-ar_t]dt + \sigma dW_t
\]

### Fonctionnalités

- construction des facteurs d’actualisation
- composants de pricing des obligations zéro-coupon
- pricing d’options sur obligations
- calcul du taux forward de swap
- calcul de l’annuité de swap
- pricing de swaptions payer
- pricing de swaptions receiver

### Calibration

Calibration sur surface synthétique de swaptions.

Paramètres calibrés :

- vitesse de retour à la moyenne
- volatilité du taux court

### Validation

- cohérence du pricing des zéro-coupons
- parité put-call sur swaptions
- récupération de paramètres sur données synthétiques
- intégration avec une courbe réelle des taux Treasury

---

# Structure du projet

```text
stochastic-model-calibration/
│
├── data/
│
├── notebooks/
│   ├── market_data.ipynb
│   ├── heston_calibration.ipynb
│   ├── sabr_calibration.ipynb
│   ├── hull_white_calibration.ipynb
│   ├── sabr_vs_heston.ipynb
│   └── real_curve_hull_white.ipynb
│
├── src/
│   ├── models/
│   ├── pricing/
│   ├── calibration/
│   ├── visualization/
│   └── utils/
│
├── tests/
│
├── report/
│
├── README.md
├── requirements.txt
└── pyproject.toml