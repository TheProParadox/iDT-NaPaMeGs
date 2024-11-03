# iDT-NaPaMeGs: Forward and Inverse Design Toolkit for Nanoparticle Metagrids

*iDT-NaPaMeGs* is an interactive application built with Streamlit for the inverse design of nanoparticle metagrid-based photonic devices. This toolkit leverages machine learning models to enable both forward and inverse design processes, optimizing MGS parameters to achieve specific optical properties, such as desired transmission and reflection spectra.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Links](#links)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About the Project

*iDT-NaPaMeGs* provides a robust framework for designing photonic devices by allowing users to specify target transmission spectra and other key parameters for nanoparticle metagrids. With a user-friendly interface powered by Streamlit, this toolkit simplifies the design process and provides both forward and inverse modeling tools to streamline prototyping and analysis in photonics and materials science.

## Features

- **Forward Modeling**: Users can input MGS parameters (height, radius, gap, and wavelength) to predict corresponding transmission (\(T_s\), \(T_p\)) and reflection (\(R_s\), \(R_p\)) spectra.
- **Inverse Design**: Users can specify target transmission and reflection values to determine optimal MGS parameters that achieve these goals.
- **Interactive Interface**: Streamlit provides an easy-to-use web-based interface for quick parameter testing and design iteration.

## Getting Started

To run this project locally, follow these steps:

### Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io) for the web interface
- Required Python packages, specified in `requirements.txt`

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com//TheProParadox/iDT-NaPaMeGs.git
   cd iDT-NaPaMeGs
   ```
   
2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:

    ```bash
    streamlit run app.py
    ```

### Usage
The app for the forward and inverse design can be directly accessed through the links provided below or via your local Streamlit server.

- Forward Design: Input parameters to predict transmission and reflection spectra.
- Inverse Design: Specify target spectra to identify optimal MGS parameters.

### Links
- Forward Design Tool: https://forward-napamegs.streamlit.app
- Inverse Design Tool: https://idt-napamegs.streamlit.app

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

This project was developed at IIT Guwahati as part of research into the design of photonic devices using metagrid-based nanostructures. Special thanks to National Supercomputing Mission (NSM) for their support.
