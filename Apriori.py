
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

data=pd.read_excel(Clustering_train.xlsx)

import Transformation
encode=Transformation.transformation(trans,'InvoiceNo','Description')

frequent_itemsets = apriori(encode, min_support = 0.01, max_len = 2, use_colnames=True)
    # compute all association rules for frequent_itemsets
Association_rule=association_rules(frequent_itemsets, metric="lift", min_threshold=1)


Association_rule['Antecedent_new'] = Association_rule.antecedents.map(lambda x : list(x)[0])
Association_rule['Consequents_new'] = Association_rule.consequents.map(lambda x : list(x)[0])

Final_Rule=Association_rule[['Antecedent_new','Consequents_new','support','confidence','lift']].sort_values('lift', axis=0, ascending=False)

