using DelimitedFiles

data = readdlm("tennis.csv", ','; skipstart = 1)

x1 = data[:, 1]
x2 = data[:, 2]
x3 = data[:, 3]
x4 = data[:, 4]

y = data[:, 5]

uniq_x1 = unique(x1)
uniq_x2 = unique(x2)
uniq_x3 = unique(x3)
uniq_x4 = unique(x4)
uniq_y = unique(y)

len_y = length(y)

len_yes = count(x -> x == "yes", y)
len_no = count(x -> x == "no", y)

p_yes = len_yes / len_y

p_no = len_no / len_y

data_yes = data[data[:, 5].=="yes", :]
data_no = data[data[:, 5].=="no", :]

len_suny_yes = count(x -> x == uniq_x1[1], data_yes)
len_overcast_yes = count(x -> x == uniq_x1[2], data_yes)
len_rainy_yes = count(x -> x == uniq_x1[3], data_yes)

len_hot_yes = count(x -> x == uniq_x2[1], data_yes)
len_mild_yes = count(x -> x == uniq_x2[2], data_yes)
len_cool_yes = count(x -> x == uniq_x2[3], data_yes)

len_high_yes = count(x -> x == uniq_x3[1], data_yes)
len_normal_yes = count(x -> x == uniq_x3[2], data_yes)

len_false_yes = count(x -> x == uniq_x4[1], data_yes)
len_true_yes = count(x -> x == uniq_x4[2], data_yes)


len_suny_no = count(x -> x == uniq_x1[1], data_no)
len_overcast_no = count(x -> x == uniq_x1[2], data_no)
len_rainy_no = count(x -> x == uniq_x1[3], data_no)

len_hot_no = count(x -> x == uniq_x2[1], data_no)
len_mild_no = count(x -> x == uniq_x2[2], data_no)
len_cool_no = count(x -> x == uniq_x2[3], data_no)

len_high_no = count(x -> x == uniq_x3[1], data_no)
len_normal_no = count(x -> x == uniq_x3[2], data_no)

len_false_no = count(x -> x == uniq_x4[1], data_no)
len_true_no = count(x -> x == uniq_x4[2], data_no)


# prediction #1: newX = ["sunny", "hot"]

p_yes_newX = (len_suny_yes / len_yes) * (len_hot_yes / len_yes) * p_yes

p_no_newX = (len_suny_no / len_no) * (len_hot_no / len_no) * p_no

p_yes_newX_n = p_yes_newX / (p_yes_newX + p_no_newX)

p_no_newX_n = p_no_newX / (p_yes_newX + p_no_newX)