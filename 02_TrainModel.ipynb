{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f47f02af",
   "metadata": {},
   "source": [
    "#### Script to learn a RF model using\n",
    "1. Scikit Learn\n",
    "2. Pycaret"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a11a9ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import ensemble\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c27e525",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_cols = ['TCB_slope',\n",
    "             'TCB_offset',\n",
    "             'TCG_slope', \n",
    "             'TCG_offset', \n",
    "             'TCW_slope',\n",
    "             'TCW_offset',\n",
    "             'NDVI_slope',\n",
    "             'NDVI_offset',\n",
    "             'NDMI_slope',\n",
    "             'NDMI_offset',\n",
    "             'NDWI_slope',\n",
    "             'NDWI_offset', \n",
    "             'elevation', \n",
    "             'slope'] \n",
    "label = 'L4'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7aa34c0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "INDATA1 = Path('train_data.csv')\n",
    "INDATA2 = Path('train_dem.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f606ce05",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1 = pd.read_csv(INDATA1)\n",
    "df2 = pd.read_csv(INDATA2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dcac85f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df1.join(df2, on=('id'), rsuffix='r_').dropna(subset=[label])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "404afdba",
   "metadata": {},
   "source": [
    "#### Define columns for analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fea22823",
   "metadata": {},
   "source": [
    "#### Create Train datasets "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93a188c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train = df[data_cols + [label]].dropna().query('L4 in (1,2,12,21)')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc074e67",
   "metadata": {},
   "source": [
    "more cleanup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2c25802",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_train[data_cols]\n",
    "y = df_train[label]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c42328c",
   "metadata": {},
   "source": [
    "### Model Training "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac676336",
   "metadata": {},
   "source": [
    "#### simple version "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20ea8cf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = ensemble.RandomForestClassifier(oob_score=True)\n",
    "model.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c754fca",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.oob_score_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6c2a05c",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z'\n",
    "joblib.dump(model, filename)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "422e22af",
   "metadata": {},
   "source": [
    "#### CV with sklearn "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0c8026a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "397cca67",
   "metadata": {},
   "source": [
    "#### Pycaret"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab597bbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pycaret.classification import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef762863",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8f19730",
   "metadata": {},
   "source": [
    "* needs to fix sklearn version\n",
    "\n",
    "https://stackoverflow.com/questions/67728802/valueerror-setting-a-random-state-has-no-effect-since-shuffle-is-false-you-sho"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1763ae58",
   "metadata": {},
   "outputs": [],
   "source": [
    "exp_clf = setup(df_train, target=label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "444cb879",
   "metadata": {},
   "outputs": [],
   "source": [
    "best = compare_models()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6f70405",
   "metadata": {},
   "source": [
    "#### Save model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebf4bdc6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
