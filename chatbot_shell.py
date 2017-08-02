#!/usr/bin/env python

import cmd

import pandas as pd
import rasa_nlu.model as rasa_model
import rasa_nlu.config as rasa_config

import chatting
import utils


class ChatbotShell(cmd.Cmd):
    prompt = '» '
    escape_chrs =  (':','!')
    intro = '\nHow can I help you?\n'

    def __init__(self, chatbot):
        super().__init__()
        self.chatbot = chatbot

    def precmd(self, line):
        if line.startswith(self.escape_chrs):
            return line[1:]
        else:
            return 'say ' + line

    def do_say(self, line):
        print()
        intent, response = self.chatbot.process_message(line)

        if intent is 'info':
            print(
                response,
                '(%s)' % intent if self.chatbot.debug else '',
                end='\n\n')
            intent = None

        if not intent:
            intent, response = self.chatbot.generate_question(line)

        if not intent:
            intent, response = self.chatbot.intent_recommendation(line)

        print(
            response,
            '(%s)' % intent if self.chatbot.debug else '',
            end='\n\n')

        if intent == "goodbye":
            return True

    def do_exit(self, line):
        return True

    def do_get(self, line):
        """Prints first var with given name from shell, chatbot, model or globals."""


        line = line.replace(' ', '_')
        for vars_ in [self, self.chatbot, self.chatbot.pref_model, globals(), chatting]:
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
    config_path = 'config_spacy.json'
    chat_data_path = 'chat_data.json'

    model_path = utils.find_latest_model()
    if not model_path:
        print('Training rasa model ...')
        utils.train_model(chat_data_path, config_path)
        model_path = utils.find_latest_model()

    print('Loading rasa model:', model_path, '...', end=' ', flush=True)
    rasa_interpreter = rasa_model.Interpreter.load(
        model_metadata = rasa_model.Metadata.load(model_path),
        config = rasa_config.RasaNLUConfig(config_path)
    )
    print('done')

    data = pd.read_table('data.csv', index_col=0)
    pref_model = chatting.PreferenceModel(data)
    chatbot = chatting.Chatbot(pref_model, rasa_interpreter)

    shell = ChatbotShell(chatbot)
    shell.cmdloop()

if __name__ == '__main__':
    main()
