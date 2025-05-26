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
        verbose_name = 'この実験ではキーボードで文字を入力しながら行います。あなたがキーボードを使えるかどうか確認するために、「2025北海道大学」と入力してください(2025は半角)。',
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
    decision_making = models.LongStringField()
    confidence = models.CharField(
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
        player.participant.vars['id_number'] = player.id_number
        player.participant.vars['gender'] = player.gender
        player.participant.vars['age'] = player.age


class Instruction(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        return {
            **{f'kind_{i+1}': task['kind'] for i, task in enumerate(C.TASKS_INFO[:4])},
            **{f'question_{i+1}': task['question'] for i, task in enumerate(C.TASKS_INFO[:4])},
            **{f'example1_{i+1}': task['example'][0] for i, task in enumerate(C.TASKS_INFO[:4])},
            **{f'example2_{i+1}': task['example'][1] for i, task in enumerate(C.TASKS_INFO[:4])},
            **{f'annotations_{i+1}': task['annotation'] for i, task in enumerate(C.TASKS_INFO[:4])},
            **{f'instruction_{i+1}': task['instruction'][0] for i, task in enumerate(C.TASKS_INFO[:4])},
        }

class Wait_Instruction(WaitPage):
    pass

class Question(Page):
    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        return player.round_number == 1 or player.participant.vars.get(f'is_finished_round_{prev}') is True

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
    form_fields = ['decision_making', 'confidence']

    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        return player.round_number == 1 or player.participant.vars.get(f'is_finished_round_{prev}') is True

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
        choice = player.decision_making
        true_false = None
        if choice == current_question['option1']:
            true_false = 1 if current_question['rank1'] < current_question['rank2'] else 0
        elif choice == current_question['option2']:
            true_false = 1 if current_question['rank2'] < current_question['rank1'] else 0
        confidence = player.confidence
        player.participant.vars[f'decision_making_round_{player.round_number}'] = player.decision_making
        player.participant.vars[f'choice_{idx}_{player.round_number}'] = {
            'question_id': current_question['question_id'],
            'round': player.round_number,
            'choice': choice,
            'true_false': true_false,
            'confidence': confidence,
            'time_spent': elapsed_time
        }


class Pre_Chat(Page):
    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        return player.round_number == 1 or player.participant.vars.get(f'is_finished_round_{prev}') is True


class Wait_Chat(WaitPage):
    @staticmethod
    def is_displayed(player):
        prev = player.round_number - 1 if player.round_number != 1 else player.round_number
        return player.round_number == 1 or player.participant.vars.get(f'is_finished_round_{prev}') is True


class Chat(Page):
    form_model = 'player'
    timeout_seconds = 60

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
        idx = player.participant.vars['current_task_index']
        nickname = player.participant.vars['nickname_map'][idx]
        message = data['message']
        idx = player.participant.vars['current_task_index']
        timestamped_message = {
            'nickname': nickname,
            'id_in_group': player.id_in_group,
            'message': message
        }
        for p in player.group.get_players():
            if f'chat_history_{idx}' not in p.participant.vars:
                p.participant.vars[f'chat_history_{idx}'] = []
            p.participant.vars[f'chat_history_{idx}'].append(timestamped_message)
            # print(f"[{p.id_in_group}] {p.participant.vars[f'chat_history_{idx}']}")
        return {0: timestamped_message}


    @staticmethod
    def vars_for_template(player):
        prev_round = player.round_number - 1 if player.round_number != 1 else player.round_number
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
        if player.round_number == 1:
            chat_history = None
        else:
            prev_players = player.in_previous_rounds()
            prev_player = prev_players[0]
            chat_history = prev_player.participant.vars.get(f'chat_history_{idx}')
            # print(f'chat_history: {chat_history}')
        return {
            'nickname': nickname,
            'decisions': decisions,
            'question': current_question['question'],
            'option1': current_question['option1'],
            'option2': current_question['option2'],
            'chat_history': chat_history
        }


class Nth_Make_Decision(Page):
    form_model = 'player'
    form_fields = ['decision_making', 'confidence']

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
        choice = player.decision_making
        true_false = None
        if choice == current_question['option1']:
            true_false = 1 if current_question['rank1'] < current_question['rank2'] else 0
        elif choice == current_question['option2']:
            true_false = 1 if current_question['rank2'] < current_question['rank1'] else 0
        confidence = player.confidence
        player.participant.vars[f'decision_making_round_{player.round_number}'] = player.decision_making
        player.participant.vars[f'choice_{idx}_{player.round_number}'] = {
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
        decisions = [p.participant.vars.get(f'decision_making_round_{p.round_number}') for p in group.get_players()]
        if all(d == decisions[0] for d in decisions):
            for p in group.get_players():
                p.participant.vars[f'is_finished_round_{p.round_number}'] = True
                # print(f"[DEBUG{p.round_number}] {p.participant.vars[f'is_finished_round_{p.round_number}']}")
        else:
            group.loop_count += 1  # 一致しなかったのでループ回数をカウント
            for p in group.get_players():
                p.participant.vars[f'is_finished_round_{p.round_number}'] = False
                # print(f"[DEBUG{p.round_number}] {p.participant.vars[f'is_finished_round_{p.round_number}']}")


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
        task_choices = []
        for task_index, task in enumerate(C.TASKS_INFO):
            kind = task['kind']
            start_index = task_index * C.NUM_PAIRS
            end_index = start_index + C.NUM_PAIRS
            correct_count = sum(
                1 for idx in range(start_index, end_index)
                if player.participant.vars.get(f'choice_{idx}', {}).get('true_false') == 1
            )
            total_questions = C.NUM_PAIRS
            task_choices.append({
                'kind': kind,
                'correct_count': correct_count,
                'total_questions': total_questions,
                'candidates': task['candidate']
            })
        total_questions = len(C.TASKS_INFO) * C.NUM_PAIRS
        total_correct_count = sum(task['correct_count'] for task in task_choices)
        reward = 200 + 10*total_correct_count
        return {
            'total_questions': len(C.TASKS_INFO) * C.NUM_PAIRS,
            'total_correct_count': total_correct_count,
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
        'ID',
        'gender', 'age',
        'order_id','questionID', 'task_id', 'kind', 'subquestionID',
        'option1', 'option2', 'rank1', 'rank2',
        'time_step', 'choice', 'true_false', 'confidence', 'time_spent'
    ]
    for player in players:
        if player.round_number == C.NUM_ROUNDS:
            for idx, task in enumerate(player.participant.vars['all_tasks']):
                choice_data = player.participant.vars.get(f'choice_{idx}', {})
                elapsed_time = player.participant.vars.get(f'elapsed_time_{idx}', {})
                yield [
                    player.participant.code,
                    player.session.code,
                    player.participant.time_started_utc,
                    player.participant.vars.get('group_id_number'),
                    player.participant.vars.get('individual_id_number'),
                    player.participant.vars.get('gender'),
                    player.participant.vars.get('age'),
                    task['order_id'],
                    task['question_id'],
                    task['task_id'],
                    task['kind'],
                    task['subquestion_id'],
                    task['option1'],
                    task['option2'],
                    task['rank1'],
                    task['rank2'],
                    choice_data.get('choice'),
                    choice_data.get('true_false'),
                    choice_data.get('confidence'),
                    elapsed_time
                ]