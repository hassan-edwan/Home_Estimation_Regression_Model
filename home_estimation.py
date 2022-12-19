## Potential Unit tests that I can make, accuracy of model unit test, if I make an interactive part then that too poten


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from redfin_data import get_and_clean_data
import numpy as np
import unittest
import pickle


def create_model():
    
    '''
    input: model
    output: expected value
    create model
    
    '''
    
    data = get_and_clean_data()[0]
    
    ## Create Initial Regression Model:
    
    reg = LinearRegression()
    labels = data['PRICE']

    train1 = data.drop(['CITY', 'PRICE', 'DAYS_ON_MARKET'], axis = 1)
    # create training split
    x_train, x_test, y_train, y_test, = train_test_split(train1, labels, test_size = 0.05, random_state = 3)
    #fit the initial model
    reg.fit(x_train, y_train)
    

    
    ## Make Adjustments to model to make it more accurate:
    model = ensemble.GradientBoostingRegressor(
        n_estimators = 3000,
        max_features= "log2",
        max_depth=6,
        min_samples_leaf = 9,
        learning_rate = 0.1,
        loss = 'huber',
        )
    

    ## fit the model
    model.fit(x_train, y_train)
   ## check accuracy
    accuracy = model.score(x_test, y_test)
    
    #save the trained model to a file
    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)
      
   ## return the the model created
    return model, accuracy

def get_model():
    
    """
    returns the model in a way that it does not need to retrain every time its called
    
    """
    
    # Load the trained model from the file
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    
    return model
    

def estimate_value(model, new_input):
    """
    returns tke estimated value
    
    """
    
    
    new_input_array = np.array(new_input).reshape(1, -1)
    new_output = model.predict(new_input_array)
    
    return new_output


class TestModel(unittest.TestCase):
    
    """
    
    unit tests
    
    def setup: sets up the parameters that the test will use to check if its right
    def test_estimate_value: returns if the estimated value is acceptably within the range of the actual property value
    def test_model_accuracy: makes sure that the accuracy is above an acceptable value
    
    """
    
    def setUp(self):
        data = get_and_clean_data()[0]
        labels = data['PRICE']
        train1 = data.drop(['CITY', 'PRICE', 'DAYS_ON_MARKET'], axis = 1)
        self.model, self.accuracy = create_model()
        self.new_input = [1, 1, 8515,4.0,3.5,3865.0,9900.0,2010.0,40.1393351,-74.6501301,2272.0,187.0,7884.0,36.4,0.074961948,2185.0,0.0]
        self.expected_output = 750000
        self.expected_accuracy = 0.7
        self.x_train, self.x_test, self.y_train, self.y_test, = train_test_split(train1, labels, test_size = 0.05, random_state = 3)
    
    def test_estimate_value(self):
        output = estimate_value(self.model, self.new_input)
        self.assertAlmostEqual(output, self.expected_output, delta=75000)
        
    def test_model_accuracy(self):
        accuracy = self.model.score(self.x_test, self.y_test)
        self.assertGreaterEqual(accuracy, self.expected_accuracy)
        
if __name__ == '__main__':
    unittest.main()
