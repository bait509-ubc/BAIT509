import numpy as np
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def plot_model(X, y, model, predict_proba = False):
    
    # Join data for plotting
    sample = (X.join(y))
    # Create a mesh for plotting
    step = (X.max() - X.min()) / 50
    x1, x2 = np.meshgrid(np.arange(sample.min()[0]-step[0], sample.max()[0]+step[0], step[0]),
                         np.arange(sample.min()[1]-step[1], sample.max()[1]+step[1], step[1]))

    # Store mesh in dataframe
    mesh_df = pd.DataFrame(np.c_[x1.ravel(), x2.ravel()], columns=['x1', 'x2'])

    # Mesh predictions
    if predict_proba:
        mesh_df['predictions'] = model.predict_proba(mesh_df[['x1', 'x2']])[:, 1]
        # Plot
        base_plot = alt.Chart(mesh_df).mark_rect(opacity=0.5).encode(
            x=alt.X('x1', bin=alt.Bin(step=step[0]), axis=alt.Axis(title=X.columns[0])),
            y=alt.Y('x2', bin=alt.Bin(step=step[1]), axis=alt.Axis(title=X.columns[1])),
            color=alt.Color('predictions', title='P(red)', scale=alt.Scale(scheme='blueorange'))
        ).properties(
            width=400,
            height=400
        )
        return alt.layer(base_plot).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        )
    else:
        mesh_df['predictions'] = model.predict(mesh_df[['x1', 'x2']])
        # Plot
        scat_plot = alt.Chart(sample).mark_circle(
            stroke='black',
            opacity=1,
            strokeWidth=1.5,
            size=100
        ).encode(
            x=alt.X(X.columns[0], axis=alt.Axis(labels=True, ticks=True, title=X.columns[0])),
            y=alt.Y(X.columns[1], axis=alt.Axis(labels=True, ticks=True, title=X.columns[1])),
            color=alt.Color(y.columns[0])
        )
        base_plot = alt.Chart(mesh_df).mark_rect(opacity=0.5).encode(
            x=alt.X('x1', bin=alt.Bin(step=step[0])),
            y=alt.Y('x2', bin=alt.Bin(step=step[1])),
            color=alt.Color('predictions', title='Legend')
        ).properties(
            width=400,
            height=400
        )
        return alt.layer(base_plot, scat_plot).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        )

def plot_regression_model(X, y, model, config=True):
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
        
    x_grid = np.linspace(min(X), max(X), 1000)
    y_predicted = np.squeeze(model.predict(x_grid))
    
    df1 = pd.DataFrame({'X': np.squeeze(X),
                        'y': np.squeeze(y)})
    df2 = pd.DataFrame({'X': np.squeeze(x_grid),
                        'y': np.squeeze(y_predicted)})
    
    scatter = alt.Chart(df1
    ).mark_circle(size=100,
                  color='red',
                  opacity=1
    ).encode(
        x='X',
        y='y')

    line = alt.Chart(df2
    ).mark_line(
    ).encode(
        x='X',
        y='y'
    )

    if config:
        return alt.layer(scatter, line).configure_axis(
                labelFontSize=20,
                titleFontSize=20
            ).configure_legend(
                titleFontSize=20,
                labelFontSize=20
            )
    else:
        return alt.layer(scatter, line)
    
def plot_knn_grid(X, y, k):
    chart1 = plot_regression_model(X, y, KNeighborsRegressor(n_neighbors=k[0]).fit(X, y), config=False).properties(title=f'k = {k[0]}')
    chart2 = plot_regression_model(X, y, KNeighborsRegressor(n_neighbors=k[1]).fit(X, y), config=False).properties(title=f'k = {k[1]}')
    chart3 = plot_regression_model(X, y, KNeighborsRegressor(n_neighbors=k[2]).fit(X, y), config=False).properties(title=f'k = {k[2]}')
    chart4 = plot_regression_model(X, y, KNeighborsRegressor(n_neighbors=k[3]).fit(X, y), config=False).properties(title=f'k = {k[3]}')
    
    return (alt.vconcat(chart1, chart3) | alt.vconcat(chart2, chart4)).configure_axis(
            labelFontSize=18,
            titleFontSize=18
        ).configure_title(
            fontSize=20
        ).configure_legend(
            titleFontSize=18,
            labelFontSize=18
        )

def plot_tree_grid(X, y, max_depth):
    chart1 = plot_regression_model(X, y, DecisionTreeRegressor(max_depth=max_depth[0]).fit(X, y), config=False).properties(title=f'max_depth = {max_depth[0]}')
    chart2 = plot_regression_model(X, y, DecisionTreeRegressor(max_depth=max_depth[1]).fit(X, y), config=False).properties(title=f'max_depth = {max_depth[1]}')
    chart3 = plot_regression_model(X, y, DecisionTreeRegressor(max_depth=max_depth[2]).fit(X, y), config=False).properties(title=f'max_depth = {max_depth[2]}')
    chart4 = plot_regression_model(X, y, DecisionTreeRegressor(max_depth=max_depth[3]).fit(X, y), config=False).properties(title=f'max_depth = {max_depth[3]}')
    
    return (alt.vconcat(chart1, chart3) | alt.vconcat(chart2, chart4)).configure_axis(
            labelFontSize=18,
            titleFontSize=18
        ).configure_title(
            fontSize=20
        ).configure_legend(
            titleFontSize=18,
            labelFontSize=18
        )

def plot_lowess(X, y, z, config=True):
    
    df1 = pd.DataFrame({'X':X,
                        'y':y})
    df2 = pd.DataFrame({'X':z[:,0],
                        'y':z[:,1]})
    scatter = alt.Chart(df1
    ).mark_circle(size=100,
                  color='red',
                  opacity=1
    ).encode(
        x='X',
        y='y')

    line = alt.Chart(df2
    ).mark_line(
    ).encode(
        x='X',
        y='y'
    )
    
    if config:
        return alt.layer(scatter, line).configure_axis(
                labelFontSize=20,
                titleFontSize=20
            ).configure_legend(
                titleFontSize=20,
                labelFontSize=20
            )
    else:
        return alt.layer(scatter, line)
    
def plot_lowess_grid(X, y, k = [1, 5, 10, 20]):
    chart1 = plot_lowess(X, y, lowess(y, X, frac=k[0]/len(X)), config=False).properties(title=f'k = {k[0]}')
    chart2 = plot_lowess(X, y, lowess(y, X, frac=k[1]/len(X)), config=False).properties(title=f'k = {k[1]}')
    chart3 = plot_lowess(X, y, lowess(y, X, frac=k[2]/len(X)), config=False).properties(title=f'k = {k[2]}')
    chart4 = plot_lowess(X, y, lowess(y, X, frac=k[3]/len(X)), config=False).properties(title=f'k = {k[3]}')
    
    return (alt.vconcat(chart1, chart3) | alt.vconcat(chart2, chart4)).configure_axis(
            labelFontSize=18,
            titleFontSize=18
        ).configure_title(
            fontSize=20
        ).configure_legend(
            titleFontSize=18,
            labelFontSize=18
        )

def plot_laplace_smoothing(X_train, y_train, X_test, y_test, alpha_vals=np.logspace(-10, 1, num=12)):
    train_err = []
    test_err = []
    for alpha in alpha_vals:
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train, y_train)
        train_err.append(1-nb.score(X_train, y_train))
        test_err.append(1-nb.score(X_test, y_test))
    
    df = pd.DataFrame({'alpha': alpha_vals,
                       'train_error': train_err,
                       'test_error': test_err}).melt(id_vars='alpha',
                                                     var_name='dataset',
                                                     value_name='error')
    lines = alt.Chart(df
    ).mark_line(
    ).encode(
        x=alt.X('alpha', axis=alt.Axis(format='e'), title='laplace smoothing (alpha)', scale=alt.Scale(type='log')),
        y=alt.Y('error', title='error rate'),
        color='dataset'
    )

    return alt.layer(lines).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        ).properties(
        width=400,
        height=300
    )

def plot_svc(x1, y1, x2, y2, model, title='SVC'):
    plt.scatter(x1, y1, marker='o', s=120, edgecolor='k')
    plt.scatter(x2, y2, marker='o', s=120, edgecolor='k')
    # create grid to evaluate model
    xx2, xx1 = np.meshgrid(np.linspace(2, 14, 30),
                           np.linspace(2, 14, 30))
    XX = np.vstack([xx1.ravel(), xx2.ravel()]).T
    Z = model.decision_function(XX).reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, lw=1, fc='none', ec='k')
    plt.title(title)

def plot_svc_grid(x1, y1, x2, y2, X, y, C = [0.001, 0.01, 0.1, 1]):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.sca(ax[0,0])
    plot_svc(x1, y1, x2, y2, SVC(C=C[0], kernel='linear').fit(X, y), "C=" + str(C[0]))
    plt.sca(ax[0,1])
    plot_svc(x1, y1, x2, y2, SVC(C=C[1], kernel='linear').fit(X, y), "C=" + str(C[1]))
    plt.sca(ax[1,0])
    plot_svc(x1, y1, x2, y2, SVC(C=C[2], kernel='linear').fit(X, y), "C=" + str(C[2]))
    plt.sca(ax[1,1])
    plot_svc(x1, y1, x2, y2, SVC(C=C[3], kernel='linear').fit(X, y), "C=" + str(C[3]))