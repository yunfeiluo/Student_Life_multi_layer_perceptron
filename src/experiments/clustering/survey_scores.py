import pandas as pd

# ---------------------------------------------------------------- 
# Independent Study, Student Stree Prediction
#
# File_name: survey_scores
# Functionality: calculate the scores of surveys
# Author: Yunfei Luo
# Start date: EST May.14th.2020
# Last update: EST May.14th.2020
# ----------------------------------------------------------------

class survey_scores:
    def __init__(self, csv_file_path, neg_scale, pos_scale, pos_term, pre_col, post_col):
        self.neg_scale = neg_scale
        self.pos_scale = pos_scale
        self.pos_term = pos_term
        self.scores = dict() # map: student_id -> {pre: score, post: score}
        df = pd.read_csv(csv_file_path)

        self.missing_data = list()

        # indexing questions
        self.questions = df.columns[2:]
        for q in self.questions:
            print(q)
        # indexing scores
        for uid in df["uid"]:
            key = int(uid[1:])
            # check if key exist
            if self.scores.get(key) != None:
                continue

            self.scores[key] = {"pre": -1, "post": -1}

            # calc scores
            rows = df.loc[df['uid'] == uid]
            row1 = rows.loc[rows['type'] == "pre"]
            row2 = rows.loc[rows['type'] == "post"]

            # if len(rows[self.questions[0]]) == 1:
            #     self.missing_data.append(key)
            #     continue
            
            pre_score = 0
            post_score = 0
            for i in range(len(self.questions)):
                pos = False
                if i+1 in self.pos_term:
                    pos = True
                
                if pre_score != None:
                    cum = -1
                    try:
                        # calc pre score
                        for j in row1[self.questions[i]]:
                            cum += 1
                            if pos:
                                pre_score += self.pos_scale[j]
                            else:
                                pre_score += self.neg_scale[j]
                    except:
                        pre_score = None
                        self.missing_data.append(key)
                    if cum < 0:
                        pre_score = None
                        self.missing_data.append(key)

                if post_score != None:
                    cum = -1
                    try:
                        # calc post score
                        for j in row2[self.questions[i]]:
                            cum += 1
                            if pos:
                                post_score += self.pos_scale[j]
                            else:
                                post_score += self.neg_scale[j]
                    except:
                        post_score = None
                        self.missing_data.append(key)
                    if cum < 0:
                        post_score = None
                        self.missing_data.append(key)

            # store scores
            self.scores[key]['pre'] = pre_score
            self.scores[key]['post'] = post_score
        
        # # check
        # for key in self.scores:
        #     print("pre score of {} is {}".format(key, self.scores[key]["pre"]))
        #     print("post score of {} is {}".format(key, self.scores[key]["post"]))
        #     print(' ')
        
        # # write to a new file
        # df_out = {pre_col: list(), post_col: list()}
        # ind = list()
        # for key in self.scores:
        #     ind.append(key)
        #     df_out[pre_col].append(self.scores[key]["pre"])
        #     df_out[post_col].append(self.scores[key]["post"])
        # df_out = pd.DataFrame(df_out, index=ind)
        # df_out.index.name = "student_id"
        
        # write to a exist file        
        df_out = pd.read_csv("src/experiments/clustering/survey/scores.csv", index_col="student_id")
        pre_scores = list()
        post_scores = list()
        has = list()
        for i, row in df_out.iterrows():
            try:
                pre_scores.append(self.scores[i]['pre'])
            except:
                pre_scores.append(None)
            try:
                post_scores.append(self.scores[i]['post'])
            except:
                post_scores.append(None)
            has.append(i)
        df_out[pre_col] = pre_scores
        df_out[post_col] = post_scores

        # add a new row
        new = list()
        for key in self.scores:
            if key not in has:
                new.append(key)
        print('new', new)
        for id_ in new:
            new_row = list()
            for i in df_out.columns:
                if i == pre_col:
                    new_row.append(self.scores[key]["pre"])
                elif i == post_col:
                    new_row.append(self.scores[key]["post"])
                else:
                    new_row.append(None)
            df_out.loc[id_] = new_row
        print(df_out)

        self.missing_data = [i for i in set(self.missing_data)]
        print('missing data', self.missing_data)
        df_out.to_csv('src/experiments/clustering/survey/scores.csv')

def calc_PSS():
    neg_scale = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
    pos_scale = {"Never": 4, "Almost never": 3, "Sometime": 2, "Fairly often": 1, "Very often": 0}
    pos_term = [4,5,7,8]
    csv_file_path = "src/experiments/clustering/survey/PerceivedStressScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_PSS", "post_PSS")

def calc_PHQ_9():
    neg_scale = {"Extremely difficult": 0, "Very difficult":0, "Somewhat difficult": 0, "Not difficult at all": 0}
    pos_scale = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
    pos_term = [i for i in range(10)]
    csv_file_path = "src/experiments/clustering/survey/PHQ-9.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_PHQ_9", "post_PHQ_9")

def calc_lonliness_scale():
    neg_scale = {}
    pos_scale = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    pos_term = [i for i in range(21)]
    csv_file_path = "src/experiments/clustering/survey/LonelinessScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_lonliness_scale", "post_longliness_scale")

def calc_flourishing_scale():
    neg_scale = {}
    pos_scale = {}
    for i in range(8):
        pos_scale[i+1] = i+1
    pos_term = [i for i in range(9)]
    csv_file_path = "src/experiments/clustering/survey/FlourishingScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_flourishing_scale", "post_flourishing_scale")

def calc_panas():
    neg_scale = {}
    pos_scale = {}
    for i in range(5):
        pos_scale[i+1] = i+1
        neg_scale[i+1] = 0
    pos_term = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
    csv_file_path = "src/experiments/clustering/survey/panas.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_panas_positive", "post_panas_positive")

if __name__ == "__main__":
    # calc_PHQ_9()
    # calc_PSS()
    # calc_lonliness_scale()
    # calc_flourishing_scale()
    # calc_panas()

    # # aggregate previous calculated scores (Big five, and some of the sleep score)
    df_out = pd.read_csv("src/experiments/clustering/survey/scores.csv", index_col="student_id")
    previous_df = pd.read_csv("Data/data/surveys_and_covariates/high_lelvel_aggregated_data.csv", index_col="student_id")
    new_col = dict() # map: col_name -> list()
    for item in previous_df.columns:
        new_col[item] = list()
    # print(df_out.loc[0, 'post_PHQ_9'])

    # extract value from previous calculated file
    has = list()
    for i, row in df_out.iterrows():
        has.append(i)
        for item in new_col:
            try:
                new_col[item].append(previous_df.loc[i, item])
            except:
                new_col[item].append(None)
    for item in new_col:
        df_out[item] = new_col[item]
    
    new = list()
    for i, row in previous_df.iterrows():
        if i not in has:
            new.append(i)
    print('new', new)

    # append new row
    for id_ in new:
        new_row = list()
        for item in df_out.columns:
            try:
                new_row.append(previous_df.loc[id_, item])
            except:
                new_row.append(None)
        df_out.loc[id_] = new_row
    print(df_out)
    
    # write to file
    df_out.to_csv('src/experiments/clustering/survey/scores.csv')
    
