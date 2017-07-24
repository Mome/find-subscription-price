#! /usr/bin/python3

import time
import cmd
from pprint import pprint

import pandas as pd
import rasa_nlu.model as rasa_model
import rasa_nlu.config as rasa_config


RASA_MODEL_ID = "20170724-154802"
DEBUG = True


# ---------------------------------------------------------------------------- #


class PreferenceModel:
    def __init__(self, data):
        self.cols = data.columns.tolist()
        self.brands = list(set(data['Brand']))
        self.categories = list(set(data['Category']))
        prices = data['Subscription Plan']
        self.price_range = (prices.min(), prices.max())
        
        self.price_pref = [None, None]
        self.category_pref = None
        self.brand_pref = [0.5]*len(self.categories)
    
    
    def __str__(self):
        return (
            f"Category: {self.category_pref}\n"
            f"Brand: {self.brand_pref}\n"
            f"Price Range: {self.price_pref}")


# ---------------------------------------------------------------------------- #


class Chatbot:
    def __init__(self, pref_model, rasa_interpreter, debug=False, confidence_minimum=0.3):
        self.pref_model = pref_model
        self.rasa_interpreter = rasa_interpreter
        self.debug = debug
        self.confidence_minimum = confidence_minimum
    
    def process_message(self, msg):
        rasa_dict = self.rasa_interpreter.parse(msg)
        
        if self.debug: print('rasa_dict:'); pprint(rasa_dict)
        
        # if confidence to low: message not understood
        if rasa_dict['intent']['confidence'] < self.confidence_minimum:
            customer_intent = 'unknown'
        else:
            customer_intent = rasa_dict['intent']['name']
        
        # calls method corresponding to intent
        try:
            identifier = 'do_' + customer_intent
            callback = vars(type(self))[identifier]
            bot_intent, response = callback(self, msg)
        except KeyError:
            print(f'No response defined for intent: {customer_intent}')
            bot_intent, response = None, None
        
        return bot_intent, response
    
    def do_goodbye(self, msg):
        return "goodbye", "bye bye"
    
    def do_unknown(self, msg):
        return "unknown", "I do not undestand!"
    
    def do_greet(self, msg):
        return "greet", "hi!"
    

# ---------------------------------------------------------------------------- #


class ChatbotShell(cmd.Cmd):
    prompt = '» '
    escape_chrs =  (':','!')
    intro = '--- enter %shelp to get help ---\n' % escape_chrs[0]
    
    def __init__(self, chatbot):
        super().__init__()
        self.chatbot = chatbot

    def precmd(self, line):
        if line.startswith(self.escape_chrs):
            return line[1:]
        else:
            return 'say ' + line
    
    def do_say(self, line):
        intent, response = self.chatbot.process_message(line)
        print('bot:', response, f'({intent})', end='\n'*2)
        if intent == "goodbye":
            return True
        
    def do_exit(self, line):
        return True
    
    def do_get(self, line):
        """Prints first var with given name from shell, chatbot, model or globals."""
        
        line = line.replace(' ', '_')
        for vars_ in [self, self.chatbot, self.chatbot.pref_model, globals()]:
            if not isinstance(vars_, dict):
                vars_ = vars(vars_)
            if line in vars_:
                print('\n', vars_[line], '\n', sep='')
                break
        else:
            print('cannot find identifier:', line)
    
    def do_debug(self, line):
        self.chatbot.debug = not self.chatbot.debug
        print('debug mode:', 'on' if self.chatbot.debug else 'off')
        
    def default(self, line):
        self.do_get(line)

        
# ---------------------------------------------------------------------------- #


def main():
    data = pd.read_table('data.csv', index_col=0)
    
    pref_model = PreferenceModel(data)
    
    print('Loading rasa model ... ', flush=True, end = '')
    rasa_interpreter = rasa_model.Interpreter.load(
        model_metadata = rasa_model.Metadata.load("./models/model_" + RASA_MODEL_ID),
        config = rasa_config.RasaNLUConfig("config_spacy.json"))
    print('done')
    
    chatbot = Chatbot(pref_model, rasa_interpreter)

    ChatbotShell(chatbot).cmdloop()

if __name__ == '__main__':
    main()