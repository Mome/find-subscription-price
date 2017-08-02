#! /usr/bin/python3

import numpy as np
import spacy

from utils import (
    render_enum,
    has_negation,
    normalize_text,
    find_numbers,
)

nlp = spacy.load('en')

synonyms= {
    "dont" : 'do not',
    'doesnt' : 'does not',
    "doesn't" : 'does not',
    "don't" : "do not",
}

category_synonyms = {
    "phone" : "Phones & Tablets",
    "computer" : "Computing",
}

brands = ['Microsoft', 'Oculus', 'Tchibo', 'Apple', 'Suunto', 
          'Polar', 'Samsung', 'Asus', 'Lenovo', 'Parrot', 'HTC', 'Amazon']
brand_synonyms = {b.lower() : b for b in brands}


# ---------------------------------------------------------------------------- #


class PreferenceModel:
    def __init__(self, data):
        self.cols = data.columns.tolist()
        self.brands = list(set(data['Brand']))
        self.categories = list(set(data['Category']))
        prices = data['Subscription Plan']
        self.price_range = (prices.min(), prices.max())
        
        self.price_pref = None
        self.category_pref = None
        self.brand_pref = {c : None for c in self.brands}
        self.brand_importance = 1.0
        self.data = data
    
    def calc_recommendations(self):
        data = self.data
        
        # select category
        if self.category_pref:
            bool_index = (data['Category'] == self.category_pref)
            data = data[bool_index]
        
        # select by price preference
        if self.price_pref:
            low, high = self.price_pref
            subsc_plan = np.array(data['Subscription Plan'])
            leq_index = np.less_equal(subsc_plan, high)
            geq_index = np.greater_equal(subsc_plan, low)
            bool_index = np.logical_and(leq_index, geq_index)
            data = data[bool_index]
    
        brand_prefs = np.array([self.brand_pref[b] for b in data['Brand']])
        bool_index = np.equal(brand_prefs, None)
        brand_prefs[bool_index] = 0.5
        prefs = self.brand_importance * brand_prefs
        
        return sorted(zip(prefs, data['Product Name']))
    
    
    
    def adjust_brand_pref(self, brand, val):
        if self.brand_pref[brand] is None:
            self.brand_pref[brand] = 0.5
        self.brand_pref[brand] *= val
    
    def __str__(self):
        var = (self.category_pref, self.brand_pref, self.price_pref)
        return (
            "Category: %s\n"
            "Brand: %s\n"
            "Price Range: %s") % var


# ---------------------------------------------------------------------------- #


class Chatbot:
    def __init__(self, pref_model, rasa_interpreter, debug=False, confidence_minimum=0.3):
        self.pref_model = pref_model
        self.rasa_interpreter = rasa_interpreter
        self.debug = debug
        self.confidence_minimum = confidence_minimum
        self.last_intent = None
        self.last_response = None
        self.expected_intent = None
        
    def process_message(self, msg):
        rasa_dict = self.rasa_interpreter.parse(msg)
        
        if self.debug:
            print('entities:', rasa_dict['entities'])
            print('intent:', rasa_dict['intent'])
            #print('intent_ranking:', rasa_dict['intent_ranking'])
            print()
        
        # if confidence to low: message not understood
        if rasa_dict['intent']['confidence'] < self.confidence_minimum:
            customer_intent = 'unknown'
        else:
            customer_intent = rasa_dict['intent']['name']
        
        # calls method corresponding to intent
        try:
            identifier = 'intent_' + customer_intent
            callback = vars(type(self))[identifier]
            bot_intent, response = callback(self, msg)
        except KeyError:
            print('No response defined for intent: ' + customer_intent)
            bot_intent, response = None, None
        
        self.last_intent = bot_intent
        self.last_response = response
        
        return bot_intent, response
    
    def generate_question(self, msg):
        if self.pref_model.category_pref is None:
            return 'category_question', 'What do you want to rent?'
        
        if not any(self.pref_model.brand_pref.values()):
            return 'brand_question', 'What brand do you prefere?'
        
        if self.pref_model.price_pref is None:
            return 'price_question', 'What price range are you looking for?'
        
        return None, None
        
    def intent_brand_pref(self, msg):
        brands_in_msg = []
        for key, val in brand_synonyms.items():
            if key in msg:
                self.pref_model.adjust_brand_pref(val, 2)
                brands_in_msg.append(val)
        
        if brands_in_msg:
            rest = ', '.join(brands_in_msg[:-1])
            response = 'OK, you like ' + render_enum(brands_in_msg) + '.'
        else:
            response = 'intet is brands_pref but didnt found any brands'
            
        return 'info', response
    
    def intent_brand_relevance(self, msg):
        return None, None
    
    def intent_category_pref(self, msg):
        for key, val in category_synonyms.items():
            if key in msg:
                self.pref_model.category_pref = val
                break
        
        if self.pref_model.category_pref:
            response = 'So you want to rent ' +  val + '.'
        else:
            response = 'intent is category_pref but no category found'
        
        return 'info', response
   
    def intent_price_pref(self, msg):
        
        if 'high' in msg:
            self.pref_model = 'high'
        
        if 'low' in msg:
            self.pref_model = 'low'
        
        if 'low' in msg:
            self.pref_model = 'low'
        
        # todo match number with regex
        numbers = find_numbers(msg)
        if numbers:
           self.pref_model.price_pref = sorted(numbers[:2])
        
        if  self.pref_model.price_pref:
            response = 'So, your price range is between %s and %s' % self.pref_model.price_pref
            return ('info', response)
         
        return (None, None)

    def intent_greet(self, msg):
        return "greet", "hi!"
    
    def intent_goodbye(self, msg):
        return "goodbye", "bye bye"
    
    def intent_unknown(self, msg):
        return "unknown", "I do not undestand!"
    
    def intent_recommendation(self, msg):
        recomms = self.pref_model.calc_recommendations()
        response = 'I recomment ' + recomms[0][1] + '.'
        return "recommendation", response
    
    def intent_question(self, msg):
        if 'brand' in msg:
            return 'answer', 'We offer ' + render_enum(self.pref_model.brands) + '.'
        
        if 'price' in msg:
            return 'answer', 'We offer between %s€ and %s€.' % self.pref_model.price_range
        
        return 'answer', 'We offer ' + render_enum(self.pref_model.categories) + '.'


