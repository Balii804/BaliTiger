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
    assert b"Predict House Price" in response.data  # Matches the title in the HTML

def test_predict_endpoint(client):
    """Test the prediction endpoint"""
    response = client.post('/predict', json={
        'area': 1600, 
        'basement': 3,  # Example: Number of bedrooms
        'garage': 12    # Example: House age in years
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'predicted_price' in data
    assert isinstance(data['predicted_price'], (int, float))
    assert data['predicted_price'] > 0  # Ensure price is positive

def test_invalid_predict_input(client):
    """Test prediction endpoint with invalid input"""
    response = client.post('/predict', json={
        'area': 1600, 
        'basement': 3
        # Missing 'garage' key
    })
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'Invalid input format'
