def conductancebasedlearningrate(conductance):
    
    Gth1 = 1
    Gth2 = 7
    if conductance <Gth1 :
        return 0

    elif conductance >=Gth2  :
        return 2.7

    else :
        return 0.2
        
    
