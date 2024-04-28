Jain = {
    'API': 'Wordline',

    #"level":"chip",
    "CoreNum"       :[4,1], 
    "CoreNoc"       :None, 
    "CoreNocCost"   :None, #{None/dis matrix}
    "GBBuf"         :None, #size, read size
    "CoreALU"       :None, #op/s
    "CoreBus"       :None, 

    #"level":"core",
    "XBNum"         :[2,1],
    "XBNoc"         :None, 
    "XBNocCost"     :None, #{None/dis matrix}
    "LCBuf"         :None, #size, read size
    "XBALU"         :None, #op/s
    'XBbus'         :None,  #

    #"level":"xb",
    "XBSize"    :[256, 64],
    "MaxRC"     :32,

    #"level":"device", 
    "Type"      :"SRAM", 
    "Precision" :1,
}