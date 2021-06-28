
# Data certification API

Le Wagon Data Science certification exam starter pack for the predictive API test.

**💡&nbsp;&nbsp;This challenge is completely independent of other challenges. It is not required to complete any other challenge in order to work on this challenge.**

## Setup

### Duplicate the repository for the API challenge

**📝&nbsp;&nbsp;Let's duplicate the repository of the API challenge.**

Go to https://github.com/lewagon/data-certification-api-movies:
- Click on `Use this template`
- Enter the repository name `data-certification-api-movies`
- Set it as **Public**
- Click on `Create repository from template`
- Click on `Code`
- Select `SSH`
- Copy the SSH URL of the repository (the format is `git@github.com:YOUR_GITHUB_NICKNAME/data-certification-api-movies.git`)

### Clone the repository for the API challenge

**📝&nbsp;&nbsp;Now we will clone your new repository.**

Open your terminal and run the following commands:

👉&nbsp;&nbsp;replace `YOUR_GITHUB_NICKNAME` with your **github nickname** and `PASTE_REPOSITORY_URL_HERE` with the SSH URL you just copied:

``` bash
cd ~/code/YOUR_GITHUB_NICKNAME
git clone PASTE_REPOSITORY_URL_HERE
cd data-certification-api-movies
```

### Look around

**💡&nbsp;&nbsp;The content of the challenge should look like this:**

``` bash
tree
```

```
.
├── Dockerfile
├── MANIFEST.in
├── Makefile
├── README.md
├── api
│   ├── __init__.py
│   └── app.py
├── exampack
│   ├── __init__.py
│   ├── data
│   ├── tests
│   │   └── __init__.py
│   ├── trainer.py
│   └── utils.py
├── model.joblib
├── notebooks
├── requirements.txt
├── scripts
│   └── exampack-run
└── setup.py
```

Open your favourite text editor and proceed with the challenge.

## API challenge

**📝&nbsp;&nbsp;In this challenge, you are provided with a trained model saved as `model.joblib`. The goal is to create an API that will predict the popularity of a movie based on its other features.**

👉&nbsp;&nbsp;You will only need to edit the code of the API in `api/app.py` 🚨

👉&nbsp;&nbsp;The package versions listed in `requirements.txt` should work out of the box with the pipelined model saved in `model.joblib`

### Install the required packages

The `requirements.txt` file lists the exact version of the packages required in order to be able to load the pipelined model that we provide.

``` bash
pip install -r requirements.txt
```

<details>
  <summary>👉&nbsp;&nbsp;If you encounter a version conflict while installing the packages 👈</summary>

  &nbsp;


In this case you will need to create a new virtual environment in order to be able to load the pipeline.

👉&nbsp;&nbsp;Only execute this commands if you encounter an issue while installing the packages 🚨

``` bash
pyenv install 3.8.6
pyenv virtualenv 3.8.6 certif
pyenv local certif
pip install -r requirements.txt
```

</details>

### Run a uvicorn server

**📝&nbsp;&nbsp;Start a `uvicorn` server in order to make sure that the setup works correctly.**

Run the server:

```bash
uvicorn api.app:app --reload
```

Open your browser at http://localhost:8000/

👉&nbsp;&nbsp;You should see the response `{ "ok": true }`

You will now be able to work on the content of the API while `uvicorn` automatically reloads your code as it changes.

### API specification

**Predict the popularity of a Spotify song**

`GET /predict`

| Parameter | Type | Description |
|---|---|---|
| original_title | string | original title of the movie |
| title | string | title of the movie in english  |
| release_date | string | release date |
| duration_min | float | duration of the movie in minutes |
| description | string | short summary of the movie|
| budget | float | budget spent to produce the movie in USD |
| original_language | string | original language |
| status | string | is the movie already released or not |
| number_of_awards_won | int | number of awards won for the movie |
| number_of_nominations | int | number of nominations |
| has_collection | int | if the movie is part of a sequel or not |
| all_genres | string | movie genres |
| top_countries | string | countries where the movie was produced (can be zero, one or many!) |
| number_of_top_productions | float | number of top production companies that produced the film if any |
| available_in_english | bool | whether the movie is available in english or not |

Returns a dictionary with the `title` of the movie, and predicted `popularity` as a float.

Example request:

```
/predict?title=Harry%20Potter&original_title=Harry%20Potter&release_date=2010-06-09&duration_min=150&description=Harry%20is%20a%20wizard%20that%20tries%20to%20save%20the%20world%20from%20crazy%20guys&budget=1000000&original_language=en&status=Released&number_of_awards_won=80&number_of_nominations=120&has_collection=1&all_genres=Fantasy,%20Family,%20Adventure&top_countries=United%20States%20of%20America,,%20United%20Kindgom&number_of_top_productions=3&available_in_english=True
```

Example response:

``` json
{
  "title": "Harry Potter",
  "popularity": 15
}
```

👉 It is your turn, code the endpoint in `api/app.py`. If you want to verify what data types the pipeline expects, have a look at the docstring of the `create_pipeline` method in `exampack/trainer.py`.

## API in production

**📝&nbsp;&nbsp;Push your API to production on the hosting service of your choice.**

<details>
  <summary>👉&nbsp;&nbsp;If you opt for Google Cloud Platform 👈</summary>

  &nbsp;


Once you have changed your `GCP_PROJECT_ID` in the `Makefile`, run the following commands to build and deploy your containerized API to Container Registry and finally Cloud Run.

</details>
