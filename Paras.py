threshold = 0.7

w_wrapper = 1.0

# parameters for Embedded feature selection
# initialization style, can be random or interval initialization
init_style = 'interval'
# if normalize is true, need to maintain
fit_normalized = True
min_reg = -1.0
max_reg = -1.0
min_loss = -1.0
max_loss = -1.0
# parameter for embed feature selection
# fitness = reg + alpha * loss
alpha = 100.0
loss = 'H' # can be binary (B) or Hingle (H) loss
reg = 'l1' # can be l0, l1, l2

