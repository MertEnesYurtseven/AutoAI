

def AprioriAssociation(dataset,min_support=0.5):
    from mlxtend.frequent_patterns import apriori
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets=apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets.to_csv("output/findings/AprioriAssociation.csv",index=False)
    return frequent_itemsets

def FpGrowthAssociation(dataset,min_support=0.6):
    from mlxtend.frequent_patterns import fpgrowth
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets=fpgrowth(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets.to_csv("output/FpGrowthAssociation.csv",index=False)
    return frequent_itemsets

def AssociationRules(frequent_itemsets,metric="confidence",min_threshold=0.7):
    from mlxtend.frequent_patterns import association_rules
    df=association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    df.to_csv("output/findings/AssociationRules.csv")
