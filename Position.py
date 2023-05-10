
w, h = image.size
# print(w)

#VERTICAL SECTIONS
v1 = 0.45
v2 = 0.675

#HORIZONTAL SECTIONS 
h1 = 0.2
h2 = 0.41



#PLOTTING VERTICAL SECTIONS
plt.axvline( x = v1*w, color = 'r', linestyle = '--')
plt.axvline( x = v2*w, color = 'r', linestyle = '--')

#PLOTTING HORIZONTAL SECTIONS
plt.axhline( y = h1*h, color = 'b', linestyle = ':')
plt.axhline( y = h2*h, color = 'b', linestyle = ':')


#GETTING THE IMAGE UPRIGHT
plt.axis([0, w, 0, h])
image = np.flipud(image)
plt.imshow(image)
plt.show()


#---------        --------      --------POSITIONING---------        ---------         ------------

df['%s'%hp_name] = np.zeros(len(df.index))
df['%s'%wp_name] = np.zeros(len(df.index))  
for b in range(0,len(df.index)):
               
    PositionPercW = df.loc[b,'%s'%(xmid_name)]/w 
          
    PositionPercH = df.loc[b,'%s'%(ymid_name)]/h
                           

# #---------        --------      --------LEFT&RIGHT---------        ---------         ------------   
    
    if PositionPercW < v1:
        Position = 'Left'
    elif PositionPercW > v2:
        Position = 'Right'
    else:
        Position = 'Middle'
    df.loc[b,'%s'%wp_name] = Position
    
#---------        --------      ------------DEPTH-----------        ---------         ------------

    if PositionPercH <= h1:
        Position = 'Very close'
    elif PositionPercH <= h2:
        Position = 'Close'
    else:
        Position = 'Far'
    df.loc[b,'%s'%hp_name] = Position

    for i in range(0, len(df.index)):
        if df.loc[i,'%s'%wp_name] == 'Left':
            if df.loc[i,'%s'%hp_name] == 'Very close':
                df.loc[i,'%s'%pos_name] = 'adjacent to the left'####
            elif df.loc[i,'%s'%hp_name] == 'Close':
                df.loc[i,'%s'%pos_name] = 'close left'####
            elif df.loc[i,'%s'%hp_name] == 'Far':
                df.loc[i,'%s'%pos_name] = 'distanced left'#####
        
        if df.loc[i,'%s'%wp_name] == 'Middle':
            if df.loc[i,'%s'%hp_name] == 'Very close':
                df.loc[i,'%s'%pos_name] = 'too close straightly infront'#####
            elif df.loc[i,'%s'%hp_name] == 'Close':
                df.loc[i,'%s'%pos_name] = 'adjacently straight infront'#####
            elif df.loc[i,'%s'%hp_name] == 'Far':
                df.loc[i,'%s'%pos_name] = 'straight infront at a distant'#####
                
        if df.loc[i,'%s'%wp_name] == 'Right':
            if df.loc[i,'%s'%hp_name] == 'Very close':
                df.loc[i,'%s'%pos_name] = 'adjacent to the right'####
            elif df.loc[i,'%s'%hp_name] == 'Close':
                df.loc[i,'%s'%pos_name] = 'close right'####
            elif df.loc[i,'%s'%hp_name] == 'Far':
                df.loc[i,'%s'%pos_name] = 'distanced right'####
                  
