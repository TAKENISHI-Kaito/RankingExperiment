from time import time
from otree.models_concrete import ChatMessage
import itertools

import numpy as np
from otree.api import *

import json
with open('./ranking_discussion/tasks_info.json') as f:
    tasks_info = json.load(f)

doc = """
Ranking Task Experiment
"""
rng = np.random.default_rng()

class C(BaseConstants):
    NAME_IN_URL = 'ranking_discussion'
    PLAYERS_PER_GROUP = None
    TASKS_INFO = tasks_info
    NUM_PAIRS = 2
    NUM_ROUNDS = 100
    NUM_TASKS = len(TASKS_INFO)

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    current_kind_index = models.IntegerField(initial=0)
    current_pair_index = models.IntegerField(initial=0)
    loop_count = models.IntegerField(initial=1)

class Player(BasePlayer):
    typing_test = models.LongStringField(
        initial = None,
        verbose_name = 'この実験ではキーボードで文字を入力しながら行います。あなたがキーボードを使えるかどうか確認するために、「2025北海道大学」と入力してください（2025は半角）。',
    )
    group_id_number = models.IntegerField(
        initial = None,
        verbose_name = 'あなたのグループID番号を入力してください（半角）。'
        )
    individual_id_number = models.IntegerField(
        initial = None,
        verbose_name = 'あなたの個人ID番号を入力してください（半角）。'
        )
    gender = models.CharField(
        initial = None,
        choices = ['男性', '女性', '回答しない'],
        verbose_name = 'あなたの性別を教えてください。',
        widget = widgets.RadioSelect()
        )
    age = models.IntegerField(
        initial = None,
        verbose_name = 'あなたの年齢を教えてください。'
        )
    first_decision_making = models.LongStringField()
    first_confidence = models.CharField(
        initial = None,
        choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        verbose_name = 'その判断にどのくらい自信がありますか？',
        widget = widgets.RadioSelect()
    )
    nth_decision_making = models.LongStringField()
    nth_confidence = models.CharField(
        initial = None,
        choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        verbose_name = 'その判断にどのくらい自信がありますか？',
        widget = widgets.RadioSelect()
    )
    chat_fields = models.LongStringField()

# FUNCTION
def creating_session(subsession: Subsession):
    if subsession.round_number == 1:
        if subsession.session.vars.get('shuffled_tasks_info') is None:
            question_id = 1
            original_tasks_info = []
            for task in C.TASKS_INFO:
                task_copy = task.copy()
                paired = list(zip(task_copy['candidate'], task_copy['ranking']))
                task_questions = []
                for sub_id, ((opt1, opt2), (r1, r2)) in enumerate(paired, start=1):
                    task_questions.append({
                        'question_id': question_id,
                        'task_id': task_copy['task'],
                        'kind': task_copy['kind'],
                        'question': task_copy['question'],
                        'subquestion_id': sub_id,
                        'option1': opt1,
                        'option2': opt2,
                        'rank1': r1,
                        'rank2': r2
                    })
                    question_id += 1
                task_copy['questions'] = task_questions
                original_tasks_info.append(task_copy)
            rng.shuffle(original_tasks_info)
            for task in original_tasks_info:
                rng.shuffle(task['questions'])
            subsession.session.vars['shuffled_tasks_info'] = original_tasks_info
        for group in subsession.get_groups():
            players = group.get_players()
            for p in players:
                task_data = []
                for task in subsession.session.vars['shuffled_tasks_info']:
                    questions = task['questions']
                    for q in questions:
                        q_copy = q.copy()
                        if rng.random() < 0.5:
                            q_copy['option1'], q_copy['option2'] = q_copy['option2'], q_copy['option1']
                            q_copy['rank1'], q_copy['rank2'] = q_copy['rank2'], q_copy['rank1']
                        task_data.append(q_copy)
                for order_id, q in enumerate(task_data, start=1):
                    q['order_id'] = order_id
                p.participant.vars['all_tasks'] = task_data
                p.participant.vars['current_task_index'] = 0
            num_tasks = len(players[0].participant.vars['all_tasks'])
            for task_index in range(num_tasks):
                shuffled_ids = rng.permutation(players)
                for i, p in enumerate(shuffled_ids):
                    if 'nickname_map' not in p.participant.vars:
                        p.participant.vars['nickname_map'] = {}
                    p.participant.vars['nickname_map'][task_index] = f'{i+1}番さん'


def not_finished_all_tasks(player):
    return player.participant.vars['current_task_index'] < len(player.participant.vars['all_tasks'])

# PAGES
class Stand_by(Page):
    form_model = 'player'
    form_fields = ['typing_test']

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1


class Demographic(Page):
    form_model = 'player'
    form_fields = ['group_id_number', 'individual_id_number', 'gender', 'age']
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.participant.vars['group_id_number'] = player.group_id_number
        player.participant.vars['individual_id_number'] = player.individual_id_number
        player.participant.vars['gender'] = player.gender
        player.participant.vars['age'] = player.age


class Instruction(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

class Wait_Instruction(WaitPage):
    pass

class Question(Page):
    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        idx = player.participant.vars['current_task_index']
        return player.round_number == 1 or (player.participant.vars.get(f'is_finished_round_{prev}') is True and idx + 1 != len(group.get_players()[0].participant.vars['all_tasks']))

    @staticmethod
    def vars_for_template(player):
        task_index = 1
        idx = player.participant.vars['current_task_index']
        current_question = player.participant.vars['all_tasks'][idx]
        current_task = player.participant.vars['all_tasks'][idx]['kind']
        current_task_info = next(task for task in C.TASKS_INFO if task['kind'] == current_task)
        return {
            'question': current_question['question'],
            'option1': current_question['option1'],
            'option2': current_question['option2'],
            'annotations': current_task_info['annotation']
        }


class First_Make_Decision(Page):
    form_model = 'player'
    form_fields = ['first_decision_making', 'first_confidence']

    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        idx = player.participant.vars['current_task_index']
        return player.round_number == 1 or (player.participant.vars.get(f'is_finished_round_{prev}') is True and idx + 1 != len(group.get_players()[0].participant.vars['all_tasks']))

    @staticmethod
    def vars_for_template(player):
        player.participant.vars['start_time'] = time()
        idx = player.participant.vars['current_task_index']
        current_question = player.participant.vars['all_tasks'][idx]
        current_kind = current_question['kind']
        return {
            'idx': idx + 1,
            'sum_questions': len(player.participant.vars['all_tasks']),
            'order_id': current_question['order_id'],
            'question_id': current_question['question_id'],
            'task_id': current_question['task_id'],
            'kind': current_question['kind'],
            'question': current_question['question'],
            'subquestion_id': current_question['subquestion_id'],
            'option1': current_question['option1'],
            'option2': current_question['option2'],
            'confidence_question': 'その判断にどのくらい自信がありますか？',
            'confidence_choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        idx = player.participant.vars['current_task_index']
        start_time = player.participant.vars.get('start_time')
        if start_time:
            elapsed_time = time() - start_time
            player.participant.vars[f'elapsed_time_{idx}'] = elapsed_time
        current_question = player.participant.vars['all_tasks'][idx]
        choice = player.first_decision_making
        true_false = None
        if choice == current_question['option1']:
            true_false = 1 if current_question['rank1'] < current_question['rank2'] else 0
        elif choice == current_question['option2']:
            true_false = 1 if current_question['rank2'] < current_question['rank1'] else 0
        confidence = player.first_confidence
        player.participant.vars[f'decision_making_round_{player.round_number}'] = player.first_decision_making
        player.participant.vars[f'choice_{idx}_{player.round_number}'] = {
            'question_id': current_question['question_id'],
            'round': player.round_number,
            'choice': choice,
            'true_false': true_false,
            'confidence': confidence,
            'time_spent': elapsed_time
        }


class Wait_Chat(WaitPage):
    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        idx = player.participant.vars['current_task_index']
        return player.round_number == 1 or (player.participant.vars.get(f'is_finished_round_{prev}') is True and idx + 1 != len(group.get_players()[0].participant.vars['all_tasks']))


class Chat(Page):
    form_model = 'player'
    # timeout_seconds = 300

    @staticmethod
    def is_displayed(player):
        if not_finished_all_tasks(player):
            if player.round_number == 1:
                return True
            else:
                prev_round = player.round_number - 1
                return player.participant.vars.get(f'is_finished_round_{prev_round}') == False
        else:
            return False

    @staticmethod
    def live_method(player, data):
        # liveSendの処理
        idx = player.participant.vars['current_task_index']
        nickname = player.participant.vars['nickname_map'][idx]
        message = data['message']
        timestamped_message = {
            'nickname': nickname,
            'id_in_group': player.id_in_group,
            'message': message
        }
        for p in player.group.get_players():
            if f'chat_history_{idx}' not in p.participant.vars:
                p.participant.vars[f'chat_history_{idx}'] = []
            p.participant.vars[f'chat_history_{idx}'].append(timestamped_message)

        # liveRecvの処理
        if player.round_number == 1:
            chat_history = None
        else:
            prev_players = player.in_previous_rounds()
            print(f'prev_players: {prev_players}')
            prev_player = prev_players[0]
            chat_history = prev_player.participant.vars.get(f'chat_history_{idx}')
        print(f'チャット履歴：\n{chat_history}')
        return {0: chat_history}


    @staticmethod
    def vars_for_template(player):
        prev_round = player.round_number - 1 if player.round_number != 1 else player.round_number
        if player.round_number == 1 or player.participant.vars.get(f'is_finished_round_{prev_round}') is True:
            decision = player.participant.vars.get(f'decision_making_round_{prev_round}')
        idx = player.participant.vars['current_task_index']
        current_question = player.participant.vars['all_tasks'][idx]
        nickname = player.participant.vars['nickname_map'][idx]
        decisions = sorted([
            {
                'nickname': p.participant.vars['nickname_map'][idx],
                'decision': p.participant.vars.get(f'decision_making_round_{prev_round}')
            }
            for p in player.group.get_players()
        ], key=lambda d: int(d['nickname'].replace('番さん', '')))
        # if player.round_number == 1:
        #     chat_history = None
        # else:
        #     prev_players = player.in_previous_rounds()
        #     prev_player = prev_players[0]
        #     chat_history = prev_player.participant.vars.get(f'chat_history_{idx}')
        #     print(f'chat_history: {chat_history}')
        return {
            'nickname': nickname,
            'decisions': decisions,
            'question': current_question['question'],
            'option1': current_question['option1'],
            'option2': current_question['option2'],
            # 'chat_history': chat_history
        }


class Nth_Make_Decision(Page):
    form_model = 'player'
    form_fields = ['nth_decision_making', 'nth_confidence']

    @staticmethod
    def is_displayed(player):
        if not_finished_all_tasks(player):
            if player.round_number == 1:
                return True
            else:
                prev_round = player.round_number - 1
                return player.participant.vars.get(f'is_finished_round_{prev_round}') == False
        else:
            return False

    @staticmethod
    def vars_for_template(player):
        player.participant.vars['start_time'] = time()
        idx = player.participant.vars['current_task_index']
        current_question = player.participant.vars['all_tasks'][idx]
        current_kind = current_question['kind']
        pair_num = sum(1 for q in player.participant.vars['all_tasks'][:idx] if q['kind'] == current_kind) + 1
        return {
            'pair_num': pair_num,
            'order_id': current_question['order_id'],
            'question_id': current_question['question_id'],
            'task_id': current_question['task_id'],
            'kind': current_question['kind'],
            'question': current_question['question'],
            'subquestion_id': current_question['subquestion_id'],
            'option1': current_question['option1'],
            'option2': current_question['option2'],
            'confidence_question': 'その判断にどのくらい自信がありますか？',
            'confidence_choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        idx = player.participant.vars['current_task_index']
        start_time = player.participant.vars.get('start_time')
        if start_time:
            elapsed_time = time() - start_time
            player.participant.vars[f'elapsed_time_{idx}'] = elapsed_time
        current_question = player.participant.vars['all_tasks'][idx]
        choice = player.nth_decision_making
        true_false = None
        if choice == current_question['option1']:
            true_false = 1 if current_question['rank1'] < current_question['rank2'] else 0
        elif choice == current_question['option2']:
            true_false = 1 if current_question['rank2'] < current_question['rank1'] else 0
        confidence = player.nth_confidence
        player.participant.vars[f'decision_making_round_{player.round_number}'] = player.nth_decision_making
        player.participant.vars[f'nth_choice_task{idx}_{player.round_number}'] = {
            'question_id': current_question['question_id'],
            'round': player.round_number,
            'choice': choice,
            'true_false': true_false,
            'confidence': confidence,
            'time_spent': elapsed_time
        }


class Wait_Decision(WaitPage):
    @staticmethod
    def is_displayed(player):
        if not_finished_all_tasks(player):
            if player.round_number == 1:
                return True
            else:
                prev_round = player.round_number - 1
                return player.participant.vars.get(f'is_finished_round_{prev_round}') == False
        else:
            return False

    @staticmethod
    def after_all_players_arrive(group):
        idx = group.get_players()[0].participant.vars['current_task_index']
        decisions = [p.participant.vars.get(f'decision_making_round_{p.round_number}') for p in group.get_players()]
        if all(d == decisions[0] for d in decisions):
            true_false = group.get_players()[0].participant.vars.get(f'nth_choice_task{idx}_{group.get_players()[0].round_number}').get('true_false')
            for p in group.get_players():
                p.participant.vars[f'task{idx}_finished'] = p.round_number
                p.participant.vars[f'task{idx}_group_choice'] = true_false
            if idx + 1 < len(group.get_players()[0].participant.vars['all_tasks']):
                for p in group.get_players():
                    p.participant.vars[f'is_finished_round_{p.round_number}'] = True
            if idx + 1 == len(group.get_players()[0].participant.vars['all_tasks']):
                for p in group.get_players():
                    p.participant.vars[f'is_finished_round_{p.round_number}'] = True
        else:
            group.loop_count += 1
            for p in group.get_players():
                p.participant.vars[f'is_finished_round_{p.round_number}'] = False


class Unanimity(Page):
    @staticmethod
    def is_displayed(player):
        current = player.participant.vars['current_task_index']
        return player.participant.vars.get(f'is_finished_round_{player.round_number}') \
            and current < len(player.participant.vars['all_tasks'])

    @staticmethod
    def vars_for_template(player):
        round_number = player.round_number
        decision = player.participant.vars.get(f'decision_making_round_{round_number}')
        return {'decision': decision}

    @staticmethod
    def before_next_page(player, timeout_happened):
        idx = player.participant.vars['current_task_index']
        player.participant.vars['current_task_index'] = idx + 1
        player.participant.vars[f'is_finished_round_{player.round_number + 1}'] = False

class Results(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player):
        task_correct_count = []
        for idx in range(len(player.participant.vars['all_tasks'])):
            true_false = player.participant.vars.get(f'task{idx}_group_choice')
            if true_false is None:
                true_false = 0
            task_correct_count.append(true_false)
        correct_count = sum(task_correct_count)
        reward = 200 + 10*correct_count
        return {
            'total_questions': len(player.participant.vars['all_tasks']),
            'correct_count': correct_count,
            'reward': reward
        }


class Finish(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS


page_sequence = [
    Stand_by,
    Demographic,
    Instruction,
    Wait_Instruction,
    Question,
    First_Make_Decision,
    Wait_Chat,
    Chat,
    Nth_Make_Decision,
    Wait_Decision,
    Unanimity,
    Results, Finish
]


def custom_export(players):
    yield [
        'participant_code', 'session_code', 'time_started_utc',
        'groupID', 'individualID', 'gender', 'age',
        'order_id','questionID', 'task_id', 'kind', 'subquestionID', 'option1', 'option2', 'rank1', 'rank2',
        'time_step', 'choice', 'true_false', 'confidence', 'time_spent'
    ]
    for p in players:
        idx = p.participant.vars['current_task_index']
        current_question = p.participant.vars['all_tasks'][idx]
        if p.participant.vars.get(f'first_choice_task{idx}_{p.round_number}'):
            choice_data = p.participant.vars.get(f'first_choice_task{idx}_{p.round_number}', {})
            time_step = 0
            yield [
                p.participant.code,
                p.session.code,
                p.participant.time_started_utc,
                p.participant.vars.get('group_id_number'),
                p.participant.vars.get('individual_id_number'),
                p.participant.vars.get('gender'),
                p.participant.vars.get('age'),
                current_question['order_id'],
                current_question['question_id'],
                current_question['task_id'],
                current_question['kind'],
                current_question['subquestion_id'],
                current_question['option1'],
                current_question['option2'],
                current_question['rank1'],
                current_question['rank2'],
                time_step,
                choice_data.get('choice'),
                choice_data.get('true_false'),
                choice_data.get('confidence'),
                choice_data.get('time_spent')
            ]
        elif p.participant.vars.get(f'nth_choice_task{idx}_{p.round_number}'):
            choice_data = p.participant.vars.get(f'nth_choice_task{idx}_{p.round_number}', {})
            if idx == 0:
                time_step = p.round_number
            else:
                time_step = p.round_number - p.participant.vars.get(f'task{idx - 1}_finished') - 1
            yield [
                p.participant.code,
                p.session.code,
                p.participant.time_started_utc,
                p.participant.vars.get('group_id_number'),
                p.participant.vars.get('individual_id_number'),
                p.participant.vars.get('gender'),
                p.participant.vars.get('age'),
                current_question['order_id'],
                current_question['question_id'],
                current_question['task_id'],
                current_question['kind'],
                current_question['subquestion_id'],
                current_question['option1'],
                current_question['option2'],
                current_question['rank1'],
                current_question['rank2'],
                time_step,
                choice_data.get('choice'),
                choice_data.get('true_false'),
                choice_data.get('confidence'),
                choice_data.get('time_spent')
            ]