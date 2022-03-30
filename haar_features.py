import numpy as np 
from enum import Enum
from intagral_image import *

#Haar Likes Features

class HaarFeaturesType(Enum):
    twoHorizontal=(2,1)
    twoVertical=(1,2)
    threeHorizontal=(3,1)
    threeVertical=(1,3)
    four=(2,2)



class HaarLikesFeatures(object):
    def __init__(self,featurType,position,width,height,threshold,parity):
        self.type=featurType
        self.topLeft=position
        self.width=width
        self.height=height
        self.threshold=threshold
        self.parity=parity
        self.bottomRight=(position[0]+width,position[1]+height)
        
    def calc_score(self,integral_image):
      white=0
      black=0
      if (self.type==HaarFeaturesType.twoHorizontal):
        white=compute_region(integral_image,self.topLeft,(int(self.topLeft[0]+self.width/2),int(self.topLeft[1]+self.height)))
        black= compute_region(integral_image,(int(self.topLeft[0]+self.width/2),self.topLeft[1]),self.bottomRight)
                
      elif(self.type==HaarFeaturesType.twoVertical):
        white=compute_region(integral_image,self.topLeft,(int(self.topLeft[0]+self.width),int(self.topLeft[1] + self.height / 2)))
        black=compute_region(integral_image,(self.topLeft[0],int(self.topLeft[1]+self.height/2)),self.bottomRight)
      elif(self.type==HaarFeaturesType. threeHorizontal):
        white = compute_region(integral_image, self.topLeft, (int(self.topLeft[0] + self.width / 3), self.topLeft[1] + self.height))
        black = compute_region(integral_image, (int(self.topLeft[0] + self.width / 3), self.topLeft[1]), (int(self.topLeft[0] + 2 * self.width / 3), self.topLeft[1] + self.height))
        white+=compute_region(integral_image, (int(self.topLeft[0] + 2 * self.width / 3), self.topLeft[1]), self.bottomRight)
      elif(self.type==HaarFeaturesType.threeVertical):
        white =compute_region(integral_image, self.topLeft, (self.bottomRight[0], int(self.topLeft[1] + self.height / 3)))
        black = compute_region(integral_image, (self.topLeft[0], int(self.topLeft[1] + self.height / 3)), (self.bottomRight[0], int(self.topLeft[1] + 2 * self.height / 3)))
        white+= compute_region(integral_image, (self.topLeft[0], int(self.topLeft[1] + 2 * self.height / 3)), self.bottomRight)
      elif(self.type==HaarFeaturesType.four):
        white=compute_region(integral_image,self.topLeft,(int(self.topLeft[0]+self.width/2),int(self.topLeft[1]+self.height/2)))
        black=compute_region(integral_image,(int(self.topLeft[0]+self.width/2),self.topLeft[1]),(self.bottomRight[0],int(self.topLeft[1]+self.height/2)))
        black+=compute_region(integral_image,(self.topLeft[0],int(self.topLeft[1]+self.height/2)),(int(self.topLeft[0]+self.width/2),self.bottomRight[1]))
        white+=compute_region(integral_image,(int(self.topLeft[0]+self.width/2),int(self.topLeft[1] + self.height / 2)), self.bottomRight)
                
      return (white-black)   
        
        
        
    def vote(self,integral_image):
      feature_score=self.calc_score(integral_image)
      if(feature_score< self.parity*self.threshold):
        return 1
      else:
        return -1


def featuresCreation(imgWidth,imgHeight,minFeaturesWidth,maxFeaturesWidth,minFeaturesHeight,maxFeaturesHeight):
    features=[]
    for feature in HaarFeaturesType:
        featureStartWidth=max(minFeaturesWidth,feature.value[0])
        for featureWidth in range(featureStartWidth,maxFeaturesWidth,feature.value[0]):
            featureStartheight=max(minFeaturesHeight,feature.value[1])
            for featureHeight in range( featureStartheight,maxFeaturesHeight,feature.value[1]):
                for x in range(imgWidth-featureWidth):
                    for y in range (imgHeight-featureHeight):
                        features.append(HaarLikesFeatures(feature,(x,y),featureWidth,featureHeight,0,1))
                        features.append(HaarLikesFeatures(feature,(x,y),featureWidth,featureHeight,0,-1))
                        
    print(str(len(features)))
    return features
            
                