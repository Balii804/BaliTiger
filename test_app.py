import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the Home Page" in response.data

def test_predict_endpoint(client):
    """Test the prediction endpoint"""
    response = client.post('/predict', json={
        'area': 1500, 
        'basement': 0, 
        'garage': 2
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'predicted_price' in data
    assert isinstance(data['predicted_price'], (int, float))