#!/usr/bin/env python 

import cmd

import pandas as pd
import rasa_nlu.model as rasa_model
import rasa_nlu.config as rasa_config

from chatbot import PreferenceModel, Chatbot
from utils import find_latest_model

DEBUG = False


class ChatbotShell(cmd.Cmd):
    prompt = '» '
    escape_chrs =  (':','!')
    intro = '\n--- enter %shelp to get help ---\n\nbot: How can I help you?' % escape_chrs[0]
    
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
        
        if intent is 'info':
            print('bot:', response, '(%s)' % intent, end='\n\n')
            intent = None
       
        if intent is None:
            intent, response = self.chatbot.generate_question(line)
        
        if not intent:
            intent, response = self.chatbot.intent_recommendation(line)
        
        print('bot:', response, '(%s)' % intent, end='\n\n')
        
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
    
    
    model_path = find_latest_model()
    print('Loading rasa model:', model_path, '...', end=' ', flush=True)
    rasa_interpreter = rasa_model.Interpreter.load(
            model_metadata = rasa_model.Metadata.load(model_path),
        config = rasa_config.RasaNLUConfig("config_spacy.json"))
    print('done')
    
    chatbot = Chatbot(pref_model, rasa_interpreter, debug=DEBUG)

    ChatbotShell(chatbot).cmdloop()

if __name__ == '__main__':
    main()
