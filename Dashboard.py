import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import  curve_fit
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.interpolate import make_interp_spline
import csv
import os
from datetime import datetime

st.set_page_config(page_title="Demo Logística",layout="wide")
st.title("Demo Interativa - Ajuste do Modelo Logístico")
st.markdown("Use os sliders para ajustar a curva logística aos dados e explorar a convergência.")


def logistica(x,L,k,x0):
    return L / (1+np.exp(-k*(x-x0)))

@st.cache_data
def load_csv(url):
    return pd.read_csv(url,sep = ';')


def prep_populacao(df,country = 'Brazil'):
    df = df.copy()
    df = df.rename(columns = {df.columns[0]:'country',df.columns[1]:"year",df.columns[2]:"value"})
    df_br = df[df['country'] == country].sort_values('year')
    df_br = df_br[['year','value']].rename(columns = {"year":"x","value":"y"})
    return df_br



#Bacteria
x_bact = np.linspace(0, 10, 150)  
y_bact = 50 / (1 + np.exp(-1.2 * (x_bact - 5))) + np.random.normal(0, 1.2, len(x_bact))
df_bact = pd.DataFrame({"x": x_bact, "y": y_bact})

#População
POP_BRASIL = r"C:\Users\Isabe\OneDrive\Documentos\Mestrado\2° Semestre\Algebra Para Ciência de Dados\Seminario\Crescimento_Populacional_Brasil.csv"

tab1,tab2,tab3,tab4 = st.tabs(["Bactérias (demo)","População (Brasil)","LM passo-a-passo","Exercício Interativo"])




with tab1:

    st.subheader("Crescimento de bactérias (Simulado)") 
    
    col1,col2,col3 = st.columns(3)
    with col1:
        L_sim = st.slider("Capacidade de suporte (L)",30.0,100.0,50.0,key = "L_sim")
    
    with col2:
        k_sim = st.slider("Taxa de crescimento (k)",0.1,3.0,1.2,key="k_sim")
    
    with col3:
        x0_sim = st.slider("Ponto de inflexão (x0)",2.0,8.0,5.0,key='x0_sim')

    
    ruido = st.slider("Nível de ruído",0.0,3.0,1.2,key = 'ruido')

    if st.button("Gerar Nova Simulação"):

        np.random.seed(int(datetime.now().timestamp()))
        y_bact_nova = L_sim/L_sim / (1 + np.exp(-k_sim * (x_bact - x0_sim))) + np.random.normal(0, ruido, len(x_bact))
        df_bact["y"] = y_bact_nova
    
    st.dataframe(df_bact.head(5))

    fig1,(ax1,ax2) = plt.subplots(1,2, figsize= (15,5))
    ax1.scatter(df_bact['x'],df_bact['y'],label = "Dados Simulados",alpha = 0.7)
    ax1.plot(np.linspace(0,10,100),logistica(np.linspace(0,10,100),L_sim,k_sim,x0_sim),'r--',label = "Curva Teórica",linewidth = 2)
    ax1.set_xlabel("Tempo")
    ax1.set_ylabel("Crescimento")
    ax1.legend()
    ax1.grid(True,linestyle='--',alpha=0.5)
    ax1.set_title("Dados Simulados vs Curva Teórica")

    y_teorico = logistica(df_bact['x'],L_sim,k_sim,x0_sim)
    residuos = df_bact['y'] - y_teorico
    ax2.hist(residuos,bins = 20,alpha = 0.7,color = 'orange',edgecolor  ='black')
    ax2.axvline(0, color='red', linestyle='--', label='Linha Zero')
    ax2.set_xlabel("Resíduos")
    ax2.set_ylabel("Frequência")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("Distribuição dos Resíduos")


    st.pyplot(fig1)

    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        rmse = np.sqrt(np.mean(residuos**2))
        st.metric("RMSE", f"{rmse:.3f}")
    with col_met2:
        r_squared = 1 - np.sum(residuos**2) / np.sum((df_bact['y'] - df_bact['y'].mean())**2)
        st.metric("R²", f"{r_squared:.3f}")
    with col_met3:
        st.metric("Coef. Variação", f"{(rmse/df_bact['y'].mean()):.3f}")




with tab2:
    df_pop = prep_populacao(load_csv(POP_BRASIL),country='Brazil')
    st.subheader("Crescimento populacional - Brasil")
    st.write("Fonte: Nações Unidas")

    col_pop1,col_pop2 = st.columns(2)
    with col_pop1:
        ano_inicio = st.slider("Ano inicial",1960,2020,1960,key="ano_inicio")
    
    with col_pop2:
        ano_fim = st.slider("Ano final",1960,2024,2024,key="ano_fim")


    df_pop_filtrado = df_pop[(df_pop['x'] >= ano_inicio) & (df_pop['x'] <= ano_fim)]

    crescimento_total = df_pop_filtrado['y'].iloc[-1] - df_pop_filtrado['y'].iloc[0]
    taxa_crescimento_anual = (df_pop_filtrado['y'].iloc[-1] / df_pop_filtrado['y'].iloc[0]) ** (1/len(df_pop_filtrado)) - 1

    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    with col_met1:
        st.metric("População Inicial", f"{df_pop_filtrado['y'].iloc[0]:,.0f}".replace(",", "."))
    with col_met2:
        st.metric("População Final", f"{df_pop_filtrado['y'].iloc[-1]:,.0f}".replace(",", "."))
    with col_met3:
        st.metric("Crescimento Total", f"{crescimento_total:,.0f}".replace(",", "."))
    with col_met4:
        st.metric("Taxa Anual", f"{taxa_crescimento_anual*100:.2f}%")
    

    fig_pop,ax_pop = plt.subplots(figsize = (12,6))
    x_filtrado = df_pop_filtrado['x'].values
    y_filtrado = df_pop_filtrado['y'].values

    if len(x_filtrado) > 3:
        x_smooth = np.linspace(x_filtrado.min(), x_filtrado.max(), 300)
        y_smooth = make_interp_spline(x_filtrado, y_filtrado)(x_smooth)
        ax_pop.plot(x_smooth, y_smooth, color="#00BFFF", linewidth=3, label="População Brasil")
    
    ax_pop.scatter(x_filtrado, y_filtrado, color='#FFD700', s=50, alpha=0.8, label='Dados Originais')

    if st.checkbox("Mostrar Ajuste Logístico",key = "pop_ajuste"):
        try:
            popt,pcov = curve_fit(logistica,x_filtrado,y_filtrado,
                                  p0=[250e6, 0.02, 2000],maxfev = 5000)
            x_future = np.linspace(x_filtrado.min(),2050,200)
            y_future = logistica(x_future, *popt)
            ax_pop.plot(x_future,y_future,'r--',linewidth=2,
                        label=f'Projeção: L={popt[0]/1e6:.1f}M')
            
            with st.expander("Detalhes do Modelo Logístico"):
                st.write(f"Capacidade de suporte (L): {popt[0]:,.0f}".replace(",", "."))
                st.write(f"Taxa de crescimento (k): {popt[1]:.4f}")
                st.write(f"Ponto de inflexão (x0): {popt[2]:.1f}")
        except:
            st.warning("Não foi possível ajustar o modelo logístico aos dados")
        
    ax_pop.set_title(f"Crescimento da População Brasileira ({ano_inicio}-{ano_fim})", fontsize=14, pad=15)
    ax_pop.set_xlabel("Ano", fontsize=11)
    ax_pop.set_ylabel("População", fontsize=11)
    ax_pop.grid(True, linestyle="--", alpha=0.3)
    ax_pop.legend()

    ax_pop.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))


    st.pyplot(fig_pop)
    
    if len(df_pop_filtrado) > 1:
        fig_taxa,ax_taxa = plt.subplots(figsize=(10,4))
        taxas = np.diff(df_pop_filtrado['y'].values) / df_pop_filtrado['y'].values[:-1] * 100
        anos_taxa = df_pop_filtrado['x'].values[1:]

        ax_taxa.bar(anos_taxa,taxas,alpha = 0.7,color = 'green',label='Taxa de Crescimento Anual')
        ax_taxa.axhline(y=np.mean(taxas), color='red', linestyle='--', 
                       label=f'Média: {np.mean(taxas):.2f}%')
        ax_taxa.set_xlabel("Ano")
        ax_taxa.set_ylabel("Taxa de Crescimento (%)")
        ax_taxa.legend()
        ax_taxa.grid(True, linestyle="--", alpha=0.3)
        ax_taxa.set_title("Taxa de Crescimento Populacional Anual")
        st.pyplot(fig_taxa)
    
    with st.expander("Ver dados detalhados"):
        st.dataframe(df_pop_filtrado.style.format({
            'x': '{:.0f}',
            'y': '{:,.0f}'
        }))


with tab3:
    st.subheader("Ajuste Logístico Iterativo - Levenberg-Marquardt")
    escolha_dados_lm = st.selectbox("Escolha os dados para LM",["Bactérias","População"])
    maxima_iteracao = st.slider("Máximo de iterações",1,50,10)
    lambda0 = st.slider("Damping inicial λ",0.001,1.0,0.01,step=0.001)

    if escolha_dados_lm == 'Bactérias':
        df_lm = df_bact
    
    else:
        df_lm = df_pop
    
    xdata = df_lm["x"].values
    ydata = df_lm['y'].values

    L_init = st.slider("L inicial",float(df_lm["y"].min()),float(df_lm["y"].max()),float(df_lm['y'].max()))
    k_init = st.slider("k inicial",-5.0,5.0,0.5)
    x0_init = st.slider("x0 inicial",float(df_lm["x"].min()),float(df_lm['x'].max()),float(df_lm["x"].median()))

    fit_lm_btn = st.button("Executar LM iterativo")

    if fit_lm_btn:
        theta = np.array([L_init,k_init,x0_init],dtype=float)
        lambda_ = lambda0
        iter_dados = []

        for it in range(maxima_iteracao):
            ypred = logistica(xdata,*theta)
            r = ydata-ypred

            #Jacobiana
            L,k,x0 = theta
            J = np.zeros((len(xdata),3))
            exp_term = np.exp(-k*(xdata-x0))
            denom = (1+exp_term)**2
            J[:,0] = (1+exp_term)
            J[:,1] = L*(xdata-x0)*exp_term/denom
            J[:,2] = -L*k*exp_term/denom

            A = J.T @ J + lambda_ * np.eye(3)
            g = J.T @ r

            try: 
                delta = la.solve(A,g)

            except la.LinAlgError:
                st.error("Matriz Singular. Ajuste Falhou")
                break
            theta = theta + delta

            rmse = np.sqrt(np.mean((ydata - logistica(xdata,*theta))**2))
            iter_dados.append([it+1, theta[0], theta[1], theta[2], rmse])

            if it > 0 and rmse > iter_dados[-2][-1]:
                lambda_ *= 10
            
            else:
                lambda_ /= 10
            
        iter_df = pd.DataFrame(iter_dados,columns=['Iteração',"L","k","x0","RMSE"])
        st.dataframe(iter_df.round(6))

        xplot = np.linspace(xdata.min(), xdata.max(), 300)
        yplot = logistica(xplot,*theta)
        fig,ax = plt.subplots(figsize = (8,4))
        ax.scatter(xdata,ydata,label = 'Dados',zorder = 3)
        ax.plot(xplot, yplot,'r--',label=f"LM Ajuste final", zorder=4)
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Crescimento")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.success(f"Ajuste LM concluído: L={theta[0]:.4g}, k={theta[1]:.4g}, x0={theta[2]:.4g}, RMSE={rmse:.4g}")
        st.pyplot(fig)


with tab4:
    st.subheader("Exercício interativo")
    st.write("Ajuste de forma manual os parâmetros usando os sliders e veja como a curva muda.")

    usuario = st.text_input("Digite seu nome:")

    L_manual = st.slider("L manual",float(df_bact["y"].min()),float(df_bact["y"].max()*3),float(df_bact["y"].max()))
    k_manual = st.slider("k manual",-5.0,5.0,0.5)
    x0_manual = st.slider("x0 manual",float(df_bact["x"].min()),float(df_bact["x"].max()),float(df_bact["x"].median()))

    #Salva os arquivos com as interações
    def salvar_interacoes(dados):

        arquivo = "LOG.csv"
        file_exists = os.path.isfile(arquivo)

        with open(arquivo, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dados.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(dados)

        
    
    #Salva as figuras
    def salvar_graficos(fig, usuario,timestamp):


        if not os.path.exists("Graficos_Salvos"):
            os.makedirs("Graficos_Salvos")

        nome_seguro = "".join(c for c in usuario if c.isalnum() or c in (' ', '-', '_')).rstrip()

        if not nome_seguro and nome_seguro != 'Anônimo':
            nome_arquivo = f"graficos_salvos/grafico_anonimo_{timestamp}.png"
        
        else:
            nome_arquivo = f"graficos_salvos/grafico_{nome_seguro}_{timestamp}.png"


        fig.savefig(nome_arquivo, dpi=200, bbox_inches='tight', facecolor='black', edgecolor='none',pad_inches = 0.3)
        plt.close(fig)
        return nome_arquivo
     





    def grafico_interativo(L_manual,k_manual,x0_manual):
        
        xplot = np.linspace(df_bact['x'].min(),df_bact['x'].max(),300)
        yplot = logistica(xplot,L_manual,k_manual,x0_manual)

        fig2,ax2 = plt.subplots(figsize = (10,6))

        fig2.patch.set_facecolor('black')
        ax2.set_facecolor('black')

        ax2.plot(xplot,yplot,label =f"Curva Ajustada",linewidth = 2.5,color = '#00FFAA',alpha = 0.8)
        ax2.scatter(df_bact['x'],df_bact['y'],label = 'Dados simulados',color = 'orange',s = 80,zorder = 5)

        ax2.set_xlabel("Tempo",color = "white",fontsize = 14,fontweight = 'bold')
        ax2.set_ylabel("Crescimento Bacteriano",color = "white",fontsize = 14,fontweight='bold')

        ax2.set_title("Ajuste Manual - Modelo Logístico",color = 'white',fontsize = 16,fontweight='bold',pad = 20)

        ax2.legend(facecolor = "black",edgecolor = "white",labelcolor = "white",fontsize = 12,
                   loc = 'upper left',bbox_to_anchor = (0.02,0.98),
                   framealpha=0.9)

        ax2.grid(True,linestyle = '--',alpha = 0.3,color='white')
        ax2.tick_params(color = 'white')

        textstr = '\n'.join([
        'Parâmetros Ajustados:',
        f'L = {L_manual:.2f}',
        f'k = {k_manual:.2f}', 
        f'x₀ = {x0_manual:.2f}'
                            ])

        props = dict(boxstyle='round', facecolor='black', alpha=0.9,
                      edgecolor='#00FFAA',linewidth=2)
        
        ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, color='white', fontfamily='monospace',
             linespacing=1.5)

        ax2.set_xlim(df_bact["x"].min() - 0.5, df_bact["x"].max() + 0.5)
        ax2.set_ylim(df_bact["y"].min() - 5, max(df_bact["y"].max(), L_manual) + 5)

        ax2.axhline(y=L_manual, color='red', linestyle=':', alpha=0.5, 
                label=f'Limite L = {L_manual:.2f}')
        
        plt.tight_layout()

        return fig2
    

    if st.button("Gerar gráficos e Salvar Interações"):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig2 = grafico_interativo(L_manual,k_manual,x0_manual)



        interacao = {
            'Timestamp':datetime.now().isoformat(),
            "Timestamp_arquivo":timestamp,
            "Identificação":usuario if usuario else "Anônimo",
            "L_value":L_manual,
            "k_value":k_manual,
            "x0_value":x0_manual,
            "user_agent":f"grafico_{usuario if usuario else 'anonimo'}_{timestamp}.png",
            'session_id': hash(str(datetime.now()))
        }

        salvar_interacoes(interacao)
        nome_arquivo_grafico = salvar_graficos(fig2, usuario if usuario else 'Anônimo', timestamp)
        st.success(f"Interação registrada! Gráfico salvo como: {os.path.basename(nome_arquivo_grafico)}")
        st.pyplot(fig2)




