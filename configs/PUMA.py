PUMA = {
    'API': 'Crossbar',

    #"level":"chip",
    "CoreNum"       :[138,1], 
    "CoreNoc"       :"mesh", 
    "CoreNocCost"   :None, #{None/dis matrix}
    "GBBuf"         :[786432,None], #size, read size
    "CoreALU"       :None, #op/s
    "CoreBus"       :384, 

    #"level":"core",
    "XBNum"     :[2,1],
    "XBNoc"     :None, 
    "XBNocCost" :None, #{None/dis matrix}
    "LCBuf"     :8192, #size, read size
    "XBALU"     :None, #op/s
    'XBbus'     :None,  #

    #"level":"xb",
    "XBSize"    :[128, 128],
    "MaxRC"     :128,

    #"level":"device", 
    "Type"      :"ReRAM", 
    "Precision" :2,
}