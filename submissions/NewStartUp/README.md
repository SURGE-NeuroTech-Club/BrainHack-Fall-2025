# NewStartup

![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)

This repository contains the source code and instructions to get the application up and running.

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Prerequisites

You will need to have Python 3.13 or newer installed on your system.

* [Python 3.13](https://www.python.org/)

### Installation

Follow these steps to set up your development environment.

1.  **Clone the repository**

    ```sh
    git clone https://github.com/Tassadaar/newstartup.git
    cd newstartup
    ```

2.  **Create and activate a virtual environment**

    * On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

    * On macOS & Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required packages**

    All necessary packages are listed in the `requirements.txt` file. Install them with the following command:
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage

To run the application, execute the main script from the root directory of the project:

```sh
python main.py