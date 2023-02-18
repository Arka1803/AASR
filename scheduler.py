import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from math import lcm

# MULTI_RATE CONTROL SWITCHING AGAINST SCHEDULE-BASED ATTACKS

class Task:
    def __init__(self, execution_time, period, task_id=None, deadline=None, offset=0):
        self.execution_time = execution_time
        self.offset = offset
        self.period = period
        self.deadline = deadline
        self.id = task_id
        self.arrivals = []
    
    def __lt__(self, other_task):
        return self.deadline < other_task.deadline
    
    def __str__(self):
        return f"T{self.id}(e={self.execution_time}, p={self.period})"

    def __repr__(self):
        return f"T{self.id}(e={self.execution_time}, p={self.period})"

    def copy(self):
        return Task(self.execution_time, self.period, self.id, self.deadline, self.offset)    

def input_tasks(num_tasks):
    task_list = []
    print("Enter `execution time` and `period` in milliseconds (collapse by a SPACE)")
    for i in range(1, num_tasks + 1):
        e, p = map(int, input(f"\tTask {i}: ").strip().split(' '))
        if e > p:
            print('schedule infeasible ... execution time > period for task', i)
            return None
        task_list.append(Task(e, p, i))
    return task_list

def create_tasks_from_file(filename):
    task_list = []
    with open(filename, 'r') as f:
        for i in range(int(f.readline().strip())):
            e, p = map(int, f.readline().strip().split(' '))
            if e > p:
                print('schedule infeasible ... execution time > period for task', i + 1)
                sys.exit()
            task_list.append(Task(e, p, i + 1))
    return task_list

def edf_scheduler(task_list, verbose=True, randomized=False):
    hyperperiod = lcm(*[task.period for task in task_list])
    task_arrivals = [[] for _ in range(hyperperiod)]
    schedule = [-1] * hyperperiod
    if verbose:
        print("EDF scheduled for hyperperiod:", hyperperiod, "ms")
    for task in task_list:
        arrivals = np.linspace(0, hyperperiod, hyperperiod // task.period + 1, dtype=int)
        if verbose:
            print("\t", task, "arrives at ", arrivals[:-1], "(ms)")
        for i in range(len(arrivals) - 1):
            new_task = task.copy()
            new_task.deadline = arrivals[i + 1]
            task_arrivals[arrivals[i]].append(new_task)
    task_set = []
    for T in range(hyperperiod):
        if randomized:
            np.random.shuffle(task_arrivals[T])
        for arrived_task in task_arrivals[T]:
            task_set.append(arrived_task)    
        curr_task = None
        min_deadline = float("inf")
        for task in task_set:
            if task.execution_time != 0 and task.deadline - T < min_deadline:
                if task.deadline < T:
                    print('EDF schedule for the task set is INFEASIBLE!')
                    sys.exit()
                curr_task = task
                min_deadline = task.deadline - T
        if curr_task:
            schedule[T] = curr_task.id
            curr_task.execution_time -= 1
        else: schedule[T] = -1
    return schedule

def calculate_all_possibile_schedule(task_list):
    print('\nCalculating Possibilities ...')
    count = 0
    all_schedules = set()
    for _ in range(1000):
        schedule = tuple(edf_scheduler(task_list, verbose=False, randomized=True))
        if schedule not in all_schedules:
            all_schedules.add(schedule)
            count += 1
    return all_schedules

def multirate_edf_schedule():
    pass

def find_attacker_tasks(task_id, task_list, schedule):
    global trusted_task_ids
    result = {'anterior': [], 'posterior': []}
    t = 0
    while t < len(schedule) - 1:
        if schedule[t] == schedule[t + 1] or schedule[t] == -1:
            t += 1
            continue
        if schedule[t] == task_id and schedule[t + 1] not in trusted_task_ids + [-1]:
            result['posterior'].append({'at': t + 1, 'id': schedule[t + 1]})
        elif schedule[t + 1] == task_id and schedule[t] not in trusted_task_ids + [-1]:
            result['anterior'].append({'at': t + 1, 'id': schedule[t]})
        t += 1
    return result
    # # 
    # attacker_ids = set()
    # for task in task_list:
    #     if task.period > task_list[task_id - 1].period and task.id not in trusted_task_ids:
    #         attacker_ids.add(task.id)
    # t = 0
    # anterior = {}
    # while t < len(schedule):
        # if schedule[t] in attacker_ids:
        #     anterior[schedule[t]] = min(anterior.get(schedule[t], float('inf')), t)
        # elif schedule[t] == task_id:
        #     for k, v in anterior.items():
        #         result['anterior'].append({'at':v, 'id': k})
        #     anterior = {}
        # else: anterior = {}
    #     t += 1
    # t = len(schedule) - 1
    # posterior = {}
    # while t >= 0:
    #     if schedule[t] in attacker_ids:
    #         posterior[schedule[t]] = min(posterior.get(schedule[t], float('inf')), t)
    #     elif schedule[t] == task_id:
    #         for k, v in posterior.items():
    #             result['posterior'].append({'at':v, 'id': k})
    #         posterior = {}
    #     else:
    #         posterior = {}
    #     t -= 1
    # return result

def bucketize_all_schedules(trusted_task_ids, task_list, write_to_file='output.txt'):
    BUCKET = {i: {'ANTERIOR': [], 'POSTERIOR': [], 'BOTH': [], 'NONE': []} for i in trusted_task_ids}
    all_schedules = calculate_all_possibile_schedule(task_list)
    print('Total Possible Schedules: ', len(all_schedules), '\n')
    for victim_id in trusted_task_ids:
        for schedule in all_schedules:
            result = find_attacker_tasks(victim_id, task_list, schedule)
            if result['anterior'] and result['posterior']:
                attack_type = 'BOTH'
                attackers = result
            else:
                if result['anterior']:
                    attack_type = 'ANTERIOR'
                    attackers = result['anterior']
                elif result['posterior']:
                    attack_type = 'POSTERIOR'
                    attackers = result['anterior']
                else:
                    attack_type = 'NONE'
                    attackers = []
            BUCKET[victim_id][attack_type].append({'schedule': schedule, 'attackers': attackers})
    
    # generate_chart(schedule, attackers=attackers, victim_task_id=, collapse=True)
    with open(write_to_file, 'w') as f:
        s = '-' * 20 + '\nBucketization SUMMARY\n' + '-' * 20
        for i, d in BUCKET.items():
            s += f"\n Victim Task {i}:"
            for k, v in d.items():
                s += f"\n   {k} = {len(v)}"
        f.writelines(s + "\n" + str(BUCKET))
    return BUCKET

def generate_chart(schedule, attackers=None, collapse=False):
    global task_list
    tasks = sorted(task_list, key=lambda t: t.period)
    n, hyp = max(schedule), len(schedule)
    figsize = (max(hyp // 10, 8), n)
    if not collapse:
        fig, axes = plt.subplots(n, 1, figsize=figsize)
        for task, ax in zip(tasks, axes):
            ax.axis([0, hyp, 0, 2])
            ax.set_yticks([])
            ax.set_ylabel(f'Task {task.id}')
            arrivals = np.linspace(0, hyp, hyp // task.period + 1, dtype=int)
            ax.set_xticks(arrivals)
    else:
        fig = plt.figure(figsize=(max(hyp // 4, 10), max(n // 2, 2)))
        ax = plt.axes()
        ax.axis([0, hyp + 5, 0, max(n - 1, 3)])
        ax.set_yticks([])
        ax.set_xticks(list(range(0, hyp + 1, 5)))

    colors = np.array(m.colormaps['plasma'].colors)
    colors = colors[np.linspace(1, len(colors), n, dtype=int) - 1]
    for i in range(n):
        curr_ax = ax if collapse else axes[i]
        for t, s in enumerate(schedule):
            if s == i + 1:
                curr_ax.add_patch(m.patches.Rectangle((t, 0), 1, 1, color=colors[i])) 
    if attackers:
        text_pos = np.zeros(hyp)
        for k, v in attackers.items():
            for d in v:
                curr_ax = ax if collapse else axes[d['id'] - 1]
                curr_ax.annotate(text=k, xy=(d['at'], 1), xytext=(d['at'] - 1, 2 + text_pos[d['at']] / 3 * collapse), arrowprops = dict(arrowstyle = "->"))
                text_pos[max(i - 5, 0): i + 5] += 1
    plt.tight_layout()
    if collapse:
        handles = [m.patches.Patch(fc=colors[i], label=f'Task {tasks[i].id}') for i in range(n)]
        plt.legend(handles=handles)
    plt.savefig('./images/schedule.jpg')
    img = cv.imread('./images/schedule.jpg')
    cv.imshow('schedule', img)

def generate_dynamic_scheduling(task_list, collapse=False, randomized=False):
    while True:
        task_list = create_tasks_from_file('input.txt')
        schedule = edf_scheduler(task_list, randomized=randomized, verbose=False)
        generate_chart(schedule, collapse=collapse)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    # n_tasks = int(input('\nEnter number of tasks: '))
    # task_list = input_tasks(n_tasks)
    task_list = create_tasks_from_file('input.txt')
    print("Task set: ", task_list)
    schedule = edf_scheduler(task_list, randomized=True)
    if schedule:
        trusted_task_ids = [1, 2]
        dynamic = False
        attackers = find_attacker_tasks(1, task_list, schedule)
        print("\nPossible Attackers for Victim Task", 1, "\n ", attackers)
        generate_chart(schedule, attackers=attackers, collapse=False)
        bucketize_all_schedules(trusted_task_ids, task_list)
        if dynamic:
            generate_dynamic_scheduling(task_list, collapse=True, randomized=True)
        else: cv.waitKey(0)
    cv.destroyAllWindows()
    