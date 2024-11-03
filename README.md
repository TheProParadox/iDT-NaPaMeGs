# iDT-NaPaMeGs: Forward and Inverse Design Toolkit for Nanoparticle Metagrids

### Maintained by Bhavik Chandna and Meghali Deka

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
    ## Inverse Design Tool
   
    ```bash
    streamlit run odod.py
    ```
    
    ## Forward Design Tool
   
   ```bash
    streamlit run forward.py
    ```

### Usage
The app for the forward and inverse design can be directly accessed through the links provided below or via your local Streamlit server.

- Forward Design: Input parameters to predict transmission and reflection spectra.
- Inverse Design: Specify target spectra to identify optimal MGS parameters.

### Links
- Forward Design Tool: https://forward-napamegs.streamlit.app
- Inverse Design Tool: https://idt-napamegs.streamlit.app

## Disclaimer
We have choosen streamlit to host our app as it is easy to use and edit the interface. It has a downside that the site becomes inactive if no user visits site for 1month and the owner of the site has to reboot it from his side. If you need to access the site and it shows error in the future, do contact **bhavikchandna@gmail.com**.

## Acknowledgments

This project was developed at IIT Guwahati as part of research into the design of photonic devices using metagrid-based nanostructures. Special thanks to National Supercomputing Mission (NSM) for their support.
