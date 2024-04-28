ISSCC = {
    'API': 'Core',

    #"level":"chip",
    "CoreNum"     :[16,1], 
    "CoreNoc"     :"Disjoint Buffer Switch", 
    "CoreNocCost" : None, #{None/dis matrix}
    "GBBuf"       :None, #size, read size
    "CoreALU"     :None, #op/s
    "CoreBus"     : None, 

    #"level":"core",
    "XBNum"     :[1,1],
    "XBNoc"     :None, 
    "XBNocCost" :None, #{None/dis matrix}
    "LCBuf"     :None, #size, read size
    "XBALU"     :None, #op/s
    'XBbus'     :None,  #

    #"level":"xb",
    "XBSize"    :[1152, 256],
    "MaxRC"     :1152,

    #"level":"device", 
    "Type"      :"SRAM", 
    "Precision" :1,
}