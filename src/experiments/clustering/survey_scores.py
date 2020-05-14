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
    def __init__(self, csv_file_path, neg_scale, pos_scale, pos_term):
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
            key = uid[1:]
            # check if key exist
            if self.scores.get(key) != None:
                continue

            self.scores[key] = {"pre": 0, "post": 0}

            # calc scores
            rows = df.loc[df['uid'] == uid]
            row1 = rows.loc[rows['type'] == "pre"]
            row2 = rows.loc[rows['type'] == "post"]

            if len(rows[self.questions[0]]) == 1:
                self.missing_data.append(key)
                continue
            
            pre_score = 0
            post_score = 0
            try: 
                for i in range(len(self.questions)):
                    pos = False
                    if i+1 in self.pos_term:
                        pos = True
                    
                    # calc pre score
                    for j in row1[self.questions[i]]:
                        if pos:
                            pre_score += self.pos_scale[j]
                        else:
                            pre_score += self.neg_scale[j]

                    # calc post score
                    for j in row2[self.questions[i]]:
                        if pos:
                            post_score += self.pos_scale[j]
                        else:
                            post_score += self.neg_scale[j]
            except:
                self.missing_data.append(key)
                continue

            # store scores
            print("pre score of {} is {}".format(key, pre_score))
            print("post score of {} is {}".format(key, post_score))
            print(' ')
            self.scores[key]['pre'] = pre_score
            self.scores[key]['post'] = post_score
        print('missing data', self.missing_data)

def calc_PSS():
    # calculate PSS score
    neg_scale = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
    pos_scale = {"Never": 4, "Almost never": 3, "Sometime": 2, "Fairly often": 1, "Very often": 0}
    pos_term = [4,5,7,8]
    csv_file_path = "src/experiments/clustering/survey/PerceivedStressScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term)

def calc_PHQ_9():
    # calculate PHQ_9 score
    neg_scale = {"Extremely difficult": 0, "Very difficult":0, "Somewhat difficult": 0, "Not difficult at all": 0}
    pos_scale = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
    pos_term = [i for i in range(10)]
    csv_file_path = "src/experiments/clustering/survey/PHQ-9.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term)

if __name__ == "__main__":
    #calc_PHQ_9()
    calc_PSS()
    
