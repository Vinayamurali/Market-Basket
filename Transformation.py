
def transformation(data,var1,var2):
    trans_data=data.groupby([var1])[var2].apply(lambda x: ','.join(list(set(x)))).reset_index()
    transcactions = list(data[var2].apply(lambda x: sorted(x.split(','))))
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    encoder = TransactionEncoder().fit(transcactions)
    onehot = encoder.transform(transcactions)
    onehot = pd.DataFrame(onehot, columns=encoder.columns_)
    return(onehot)

    
