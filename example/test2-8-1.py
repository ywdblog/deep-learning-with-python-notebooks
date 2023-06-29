
# SGD 变体的一个实现

''' 
past_velocity=0
momentum = 0.1 
while loss>0.1:
    w,loss,gradient = get_current_parameters()
    welocity = past_velocity * momentum - learning_rate * gradient 
    w= w + momentum* welocity -  learning_rate * gradient 
    past_velocity = welocity
    update_paramter(w)
'''