# Google Account Storage Manager

Collaborators: Avni Mishra, Jasmin Duong, Rebecca Choi, Jayden Ngo


The google account storage manager is a tool that scrapes your emails and photos and tells you what items to delete so you can manage your storage without painstakingly comb through every single file in your gmail and google photos app. 


## Userwork Flow 

TODO 

## The Components

### Models 

#### Email 

TODO

#### Photo 

TODO 

### Client 

We used the Google Cloud API to set up a more seemless user workflow. 

<b>The email client</b> scrapes a users emails with an adjustable param for how many it scrapes at a time. It then formats those emails properly for the model and sends the emails into the model data pipeline. It then takes the result of the model (all the classified emails) and applies a label to those emails in your inbxo called "Selected for Deletion" The user can then view only those labelled emails and delete them.

<b>The photo client</b> is unfortunetly less useful, since there is no client that can scrape a users photo. Instead the user must manually select photos OR use google takeout and then port that into model themselves. We opted to stick with the google takeout route, since the idea of the product is the reduce the amount of time a user has to interact with their own photos. The photo client also cannot sort photos into a folder but can only add photos to your account and sort them into a folder, defeating the purpose of a storage manager entirely. Because of this we opted to not include the photos client and just interface with the model via folders on our local disk.  