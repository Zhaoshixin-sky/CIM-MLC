# example architecture
ArchTem = {
    'API': 'Wordline',

    #"level":"chip",
    "CoreNum"       :[4,4], 
    "CoreNoc"       :"mesh", 
    "CoreNocCost"   :None, # {None/dis matrix}
    "GBBuf"         :[786432,32], # size, read size
    "CoreALU"       :1024, # op/s
    "CoreBus"       :48, 

    #"level":"core",
    "XBNum"     :[4,4],
    "XBNoc"     :"mesh", 
    "XBNocCost" :None, #{None/dis matrix}
    "LCBuf"     :[1024,32], #size, read size
    "XBALU"     :1024, #op/s
    'XBbus'     :1024,  #

    #"level":"xb",
    "XBSize"    :[128, 128],
    "MaxRC"     :64,

    #"level":"device", 
    "Type"      :"F", 
    "Precision" :1,
}
