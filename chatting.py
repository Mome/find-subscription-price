#! /usr/bin/python3

from random import choice

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

# this is neccessary, because ner-detection of rsa_nlu is not reliable
categories = ['Gaming & VR', 'Smart Home', 'Computing', 'Wearables', 'Phones & Tablets', 'Drones']
category_synonyms = {c.lower() : c for c in categories}
category_synonyms.update({
    "phone" : "Phones & Tablets",
    "computer" : "Computing",
    "tablet" : "Phones & Tablets",
    "gaming" : "Gaming & VR",
    "vr" : "Gaming & VR",
    "drone" : "Drones",
    "watch" : 'Wearables',
    "smartphone" : 'Phones & Tablets',
    "vacuum" : "Smart Home",
})

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
        self.data = data

    @property
    def possible_brands(self):
        return set(self.filtered_data['Brand'])

    @property
    def possible_categories(self):
        return set(self.filtered_data['Category'])

    @property
    def possible_price_range(self):
        prices = self.filtered_data['Subscription Plan']
        return min(prices), max(prices)

    @property
    def filtered_data(self):
        """Gets subset of the data that is still possible."""

        data = self.data

        # filter category
        if self.category_pref:
            index = (data['Category'] == self.category_pref)
            data = data[index]

        # filter brands
        brands = {k for k,v in self.brand_pref.items() if (v != 0)}
        index = (data['Brand'].isin(brands))
        data = data[index]

        return data


    def calc_recommendations(self):
        data = self.filtered_data

        brand_prefs = np.array([self.brand_pref[b] for b in data['Brand']])
        bool_index = np.equal(brand_prefs, None)
        brand_prefs[bool_index] = 0.5
        prefs = np.array(brand_prefs)

        if self.price_pref:
            price_diff = data['Subscription Plan'] - self.price_pref
            price_diff = np.abs(price_diff)
            prefs /= price_diff

        #print('prefs:', prefs)

        return sorted(zip(prefs, data['Product Name']), reverse=True)

    def adjust_brand_pref(self, brand, val):
        if self.brand_pref[brand] is None:
            self.brand_pref[brand] = 0.5
        self.brand_pref[brand] *= val

        for brand, val in self.brand_pref.items():
            if val is None:
                self.brand_pref[brand] = 0.0


    def __str__(self):
        var = (self.category_pref, self.brand_pref, self.price_pref)
        return (
            "Category: %s\n"
            "Brand: %s\n"
            "Price Range: %s") % var


# ---------------------------------------------------------------------------- #


class Chatbot:
    def __init__(self, pref_model, rasa_interpreter, debug=False, confidence_minimum=0.3, expectation_boost=1.5):
        self.pref_model = pref_model
        self.rasa_interpreter = rasa_interpreter
        self.debug = debug
        self.confidence_minimum = confidence_minimum
        self.expectation_boost = expectation_boost
        self.expected_intent = None

    def process_message(self, msg):
        msg = msg.lower()
        rasa_dict = self.rasa_interpreter.parse(msg)

        if self.debug:
            print('entities:', rasa_dict['entities'])
            print('intent:', rasa_dict['intent'])
            print('intent_ranking:', rasa_dict['intent_ranking'])
            print()

        customer_intent = rasa_dict['intent']['name']
        confidence = rasa_dict['intent']['confidence']

        # account for expected intent
        if self.expected_intent:
            ranking = {d['name']:d['confidence'] for d in rasa_dict['intent_ranking']}
            boosted_confidence = ranking[self.expected_intent] * self.expectation_boost
            if boosted_confidence > confidence:
                customer_intent = self.expected_intent
                confidence = boosted_confidence

        # if confidence to low: message not understood
        if confidence < self.confidence_minimum:
            customer_intent = 'unknown'

        # calls method corresponding to intent
        try:
            identifier = 'intent_' + customer_intent
            callback = vars(type(self))[identifier]
            bot_intent, response = callback(self, msg)
        except KeyError:
            print('No response defined for intent: ' + customer_intent)
            bot_intent, response = (None, None)

        if bot_intent != 'question':
            self.expected_intent = None

        return bot_intent, response

    def generate_question(self, msg):
        if (not self.pref_model.category_pref
            and len(self.pref_model.possible_categories) > 1
            ):
            self.expected_intent = 'category_pref'
            return 'question', 'What do you want to rent?'

        if (not any(self.pref_model.brand_pref.values())
            and len(self.pref_model.possible_brands) > 1
            ):
            self.expected_intent = 'brand_pref'
            question = 'What brand do you prefere?'
            question += '\nWe have ' + render_enum(self.pref_model.possible_brands)
            return 'question', question

        if not self.pref_model.price_pref:
            low, high =  self.pref_model.possible_price_range
            if str(low) != str(high):
                self.expected_intetn = 'price_pref'
                response = 'We have offers between %s and %s.\n' % (low, high)
                response += 'What do you have in mind ?'
                return 'question', response


        return (None, None)

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

        # todo match number with regex
        numbers = find_numbers(msg)
        if numbers:
            numbers = np.array(numbers[:2], dtype=float)
            price = sum(numbers) / len(numbers)
            self.pref_model.price_pref = price
            return 'info', 'I assume a price around: %s' % price

        #if self.pref_model.price_pref:
        #    response = 'So, your price range is between %s and %s' % self.pref_model.price_pref
        #    return ('info', response)

        return (None, None)

    def intent_greet(self, msg):
        phrases = "hi!", 'hello', 'hey'
        return "greet", choice(phrases)

    def intent_goodbye(self, msg):
        phrases = 'bye bye', 'bye', 'goodbye'
        return 'goodbye', choice(phrases)

    def intent_unknown(self, msg):
        phrases = 'I do not undestand!', 'What?', 'Can you phrase that differently?'
        return 'unknown', choice(phrases)

    def intent_recommendation(self, msg):
        recomms = self.pref_model.calc_recommendations()
        response = 'I recomment the ' + recomms[0][1] + '.'
        return "recommendation", response

    def intent_question(self, msg):
        """Answers a question."""

        if self.expected_intent:
            intent = self.expected_intent
        else:
            if 'price' in msg:
                intent = 'price_pref'
            elif 'brand' in msg:
                intent = 'brand_pref'
            else:
                intent = 'category_pref'

        if intent == 'brand_pref':
            brands = self.pref_model.possible_brands
            category = self.pref_model.category_pref
            if category:
                answer = 'For %s we offer %s.' % (category, render_enum(brands))
            else:
                answer = 'We offer ' + render_enum(brands) + '.'

        if intent == 'price_pref':
            answer = 'We offer between %s€ and %s€.' % self.pref_model.price_range

        if intent == 'category_pref':
            answer = 'We offer ' + render_enum(self.pref_model.categories) + '.'

        return 'answer', answer
