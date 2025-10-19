Installation
============

Requirements
------------

Eclipsera requires:

* Python 3.8 or higher
* NumPy
* SciPy
* matplotlib (optional, for visualization)

Installing with pip
-------------------

The easiest way to install Eclipsera is using pip:

.. code-block:: bash

    pip install eclipsera

This will install Eclipsera and its core dependencies (NumPy and SciPy).

Installing from Source
----------------------

To install the latest development version from source:

.. code-block:: bash

    git clone https://github.com/tiverse/eclipsera.git
    cd eclipsera
    pip install -e .

This installs Eclipsera in "editable" mode, which is useful for development.

Verifying Installation
----------------------

To verify that Eclipsera is installed correctly:

.. code-block:: python

    import eclipsera
    print(eclipsera.__version__)

You should see the version number (e.g., "1.1.0").

Quick Test
----------

Run a quick test to ensure everything works:

.. code-block:: python

    import numpy as np
    from eclipsera.ml import LogisticRegression

    # Create sample data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    # Train model
    clf = LogisticRegression()
    clf.fit(X, y)

    # Make predictions
    predictions = clf.predict(X)
    print(f"Predictions shape: {predictions.shape}")
    print("Installation successful!")

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure NumPy and SciPy are installed:

.. code-block:: bash

    pip install numpy scipy

Version Conflicts
~~~~~~~~~~~~~~~~~

If you have version conflicts, create a fresh virtual environment:

.. code-block:: bash

    python -m venv eclipsera_env
    source eclipsera_env/bin/activate  # On Windows: eclipsera_env\\Scripts\\activate
    pip install eclipsera

Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows:**

Make sure you have Microsoft Visual C++ 14.0 or greater installed for building dependencies.

**macOS:**

If you encounter issues with SciPy, install it via Homebrew first:

.. code-block:: bash

    brew install openblas
    pip install scipy

**Linux:**

Install BLAS and LAPACK development files:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install libblas-dev liblapack-dev

    # Fedora/RHEL
    sudo yum install blas-devel lapack-devel

Development Installation
------------------------

For contributing to Eclipsera, install with development dependencies:

.. code-block:: bash

    git clone https://github.com/tiverse/eclipsera.git
    cd eclipsera
    pip install -e .
    pip install pytest black isort mypy  # Development tools

Run tests to verify:

.. code-block:: bash

    pytest tests/

Next Steps
----------

* Read the :doc:`quickstart` guide
* Explore the :doc:`user_guide`
* Check out :doc:`tutorials/classification`
