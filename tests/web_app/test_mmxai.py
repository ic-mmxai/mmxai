import pytest
from flask import Flask, url_for
from app import app as mmxai
import io

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = mmxai
    return app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_hateful_memes():
    pass

def test_home(client):
    url = '/'
    response = client.get(url)
    assert response.status_code == 200

def test_brefore_request_when_user_exists(client):
    url = '/'
    with client.session_transaction() as session:
        session['user'] = '5fRorZuF'
    response = client.get(url)
    assert response.status_code == 200

def test_docs(client):
    url = '/docs/'
    response = client.get(url)
    assert response.status_code == 200

def test_hateful_memes(client):
    url = '/explainers/hateful-memes'
    response = client.get(url)
    assert response.status_code == 200

    with client.session_transaction() as session:
        session["modelPath"] = "./static/long_filename_test.ckpt"
    response = client.get(url)
    assert response.status_code == 200

    with client.session_transaction() as session:
        session["userModel"] = "no_model"
    response = client.get(url)
    assert response.status_code == 200

    with client.session_transaction() as session:
        session["userModel"] = "mmf"
    response = client.get(url)
    assert response.status_code == 200

    with client.session_transaction() as session:
        session["userModel"] = "onnx"
    response = client.get(url)
    assert response.status_code == 200

def test_upload_image(client):
    url = '/uploadImage'
    data = {}
    response = client.post(url, data=data)
    assert response.status_code == 302

    data["inputImg"] = (io.BytesIO(b"test file"), "test.jpg")
    response = client.post(url, data=data)
    assert response.status_code == 302

    with open("01243.png", "rb") as img:
        imgBytesIO = io.BytesIO(img.read())
    data["inputImg"] = (imgBytesIO, "01243.png")
    response = client.post(url, data=data)
    assert response.status_code == 302

def test_upload_internal_model(client):
    url = '/uploadModel'
    data = {}
    response = client.post(url, data=data)
    assert response.status_code == 302

    data["selectOption"] = "internalModel"
    data["selectModel"] = "MMBT"
    response = client.post(url, data=data)
    assert response.status_code == 302
    
    data["inputCheckpoint1"] = (io.BytesIO(b"test file"), 'test.jpg')
    # with open("logo.png", 'rb') as img:
    #     imgBytesIO = io.BytesIO(img.read())
    # data["inputCheckpoint1"] = (imgBytesIO, 'logo.png')
    response = client.post(url, data=data)
    assert response.status_code == 302

    data["inputCheckpoint1"] = (io.BytesIO(b"test file"), 'test.ckpt')
    response = client.post(url, data=data)
    assert response.status_code == 302

def test_upload_self_model(client):
    url = '/uploadModel'
    data = {}

    data["selectOption"] = "selfModel"
    response = client.post(url, data=data)
    assert response.status_code == 302

    data["inputCheckpoint2"] = (io.BytesIO(b"test file"), 'test.ckpt')
    response = client.post(url, data=data)
    assert response.status_code == 302

    data["inputCheckpoint2"] = (io.BytesIO(b"test file"), 'test.onnx')
    response = client.post(url, data=data)
    assert response.status_code == 302


def test_upload_no_model(client):
    url = '/uploadModel'
    data = {}

    data["selectOption"] = "noModel"
    response = client.post(url, data=data)
    assert response.status_code == 400

    data["selectOption"] = "noModel"
    data["selectExistingModel"] = "MMBT"
    response = client.post(url, data=data)
    assert response.status_code == 302


def test_inpaint(client):
    url = '/inpaint'
    response = client.get(url)
    assert response.status_code == 302

    with client.session_transaction() as session:
        session["imgName"] = "01243.png"
    response = client.get(url)
    assert response.status_code == 302

    with client.session_transaction() as session:
        session["imgName"] = "../01243.png" #TODO: change img path to test folder
    response = client.get(url)
    assert response.status_code == 302


def test_restore(client):
    url = '/restoreImage'
    response = client.get(url)
    assert response.status_code == 302

    with client.session_transaction() as session:
        session["imgName"] = "01243_inpainted.png"
    response = client.get(url)
    assert response.status_code == 302


def test_predict(client):
    url = '/explainers/hateful-memes/predict'
    data = {}
    data["texts"] = 'how I want to say hello'
    data["expMethod"] = 'lime'
    data["expInc"] = "encourage"
    data["shapAlgo"] = "partition"
    data["shapMaxEval"] = int(50)
    data["shapBatchSize"] = int(50)
    data["limeSamplesNumber"] = int(50)
    data["torchrayMaxIteration"] = int(200)
    with client.session_transaction() as session:
        session['user'] = 'mockuser'
        session["userModel"] = "no_model"
        session["imgName"] = "../01245.png"
        session["modelType"] = "MMBT"
        session["modelPath"] = None
    response = client.post(url, data=data)
    assert response.status_code == 302
    assert 'text/html' in response.content_type



def test_predict_lime(client):
     url = '/explainers/hateful-memes/predict'
     data = {}
     data["texts"] = 'how I want to say hello'
     data["expMethod"] = 'lime'
     data["expInc"] = "encourage"
     data["shapAlgo"] = "partition"
     data["shapMaxEval"] = int(50)
     data["shapBatchSize"] = int(50)
     data["limeSamplesNumber"] = int(50)
     data["torchrayMaxIteration"] = int(200)
     with client.session_transaction() as session:
         session['user'] = 'mockuser'
         session["userModel"] = "no_model"
         session["imgName"] = "../01245.png"
         session["modelType"] = "MMBT"
         session["modelPath"] = None
     response = client.post(url, data=data)
     assert response.status_code == 302
     assert 'text/html' in response.content_type


def test_predict_shap(client):
    url = '/explainers/hateful-memes/predict'
    data = {}
    data["texts"] = 'how I want to say hello'
    data["expMethod"] = 'shap'
    data["expInc"] = "encourage"
    data["shapAlgo"] = "partition"
    data["shapMaxEval"] = int(50)
    data["shapBatchSize"] = int(50)
    data["limeSamplesNumber"] = int(50)
    data["torchrayMaxIteration"] = int(200)
    with client.session_transaction() as session:
        session['user'] = 'mockuser'
        session["userModel"] = "no_model"
        session["imgName"] = "../01245.png"
        session["modelType"] = "MMBT"
        session["modelPath"] = None
    response = client.post(url, data=data)
    assert response.status_code == 302
    assert 'text/html' in response.content_type


def test_predict_torchray(client):
    url = '/explainers/hateful-memes/predict'
    data = {}
    data["texts"] = 'how I want to say hello'
    data["expMethod"] = 'torchray'
    #data["expDir"] = "encourage"
    data["expInc"] = "encourage"
    data["shapAlgo"] = "partition"
    data["shapMaxEval"] = int(50)
    data["shapBatchSize"] = int(50)
    data["limeSamplesNumber"] = int(50)
    data["torchrayMaxIteration"] = int(200)
    with client.session_transaction() as session:
        session['user'] = 'mockuser'
        session["userModel"] = "no_model"
        session["imgName"] = "../01245.png"
        session["modelType"] = "MMBT"
        session["modelPath"] = None
    response = client.post(url, data=data)
    assert response.status_code == 302
    assert 'text/html' in response.content_type

def test_predict_default(client):
    url = '/explainers/hateful-memes/predict'
    data = {}
    data["texts"] = 'how I want to say hello'
    data["expMethod"] = 'default'
    #data["expDir"] = "encourage"
    data["expInc"] = "encourage"
    data["shapAlgo"] = "partition"
    data["shapMaxEval"] = int(50)
    data["shapBatchSize"] = int(50)
    data["limeSamplesNumber"] = int(50)
    data["torchrayMaxIteration"] = int(200)
    with client.session_transaction() as session:
        session['user'] = 'mockuser'
        session["userModel"] = "no_model"
        session["imgName"] = "../01245.png"
        session["modelType"] = "MMBT"
        session["modelPath"] = None
    response = client.post(url, data=data)
    assert response.status_code == 302
    assert 'text/html' in response.content_type

def test_predict_text_error_handling(client):
    url = '/explainers/hateful-memes/predict'
    data = {}
    data["texts"] = "error with ?"
    with client.session_transaction() as session:
        session['user'] = 'mockuser'
        session["userModel"] = "no_model"
        session["imgName"] = "../01245.png"
        session["modelType"] = "MMBT"
        session["modelPath"] = None
    response = client.post(url, data=data)
    assert response.status_code == 302
    assert 'text/html' in response.content_type

def test_predict_model_built_error_handling(client):
    url = '/explainers/hateful-memes/predict'
    data = {}
    data["texts"] = 'how I want to say hello'
    data["expMethod"] = 'mockerror'
    data["expInc"] = "mockerror"
    data["shapAlgo"] = "mockerror"
    data["shapMaxEval"] = int(50)
    data["shapBatchSize"] = int(50)
    data["limeSamplesNumber"] = int(50)
    data["torchrayMaxIteration"] = int(200)
    with client.session_transaction() as session:
        session['user'] = 'mockerror'
        session["userModel"] = "mockerror"
        session["imgName"] = "mockerror"
        session["modelType"] = "mockerror"
        session["modelPath"] = "mockerror"
    response = client.post(url, data=data)
    assert response.status_code == 302
    assert 'text/html' in response.content_type

def test_select_example(client):
    url = '/selectExample'
    data = {}
    data["exampleID"] = "10398"
    response = client.post(url, data=data)
    assert response.status_code == 302

def test_fetch_example(client):
    url = '/fetchExample'
    data = {}
    data["expMethod"] = "shap"
    response = client.post(url, data=data)
    assert response.status_code == 302

    with client.session_transaction() as session:
        session["exampleID"] = "10398"
    response = client.post(url, data=data)
    assert response.status_code == 302
