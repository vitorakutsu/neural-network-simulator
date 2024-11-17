import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate, activation='logistic'):
        self.w_hidden = np.random.randn(n_inputs, n_hidden) * 0.01
        self.b_hidden = np.zeros((1, n_hidden))
        self.w_output = np.random.randn(n_hidden, n_outputs) * 0.01
        self.b_output = np.zeros((1, n_outputs))
        self.learning_rate = learning_rate
        self.activation = activation

    def activation_function(self, x, derivative=False):
        if self.activation == 'linear':
            return 1 / 10 if derivative else x / 10
        elif self.activation == 'logistic':
            fx = 1 / (1 + np.exp(-x))
            return fx * (1 - fx) if derivative else fx
        elif self.activation == 'tanh':
            return 1 - np.power(np.tanh(x), 2) if derivative else np.tanh(x)

    def forward(self, X):
        self.hidden_net = np.dot(X, self.w_hidden) + self.b_hidden
        self.hidden_output = self.activation_function(self.hidden_net)
        self.output_net = np.dot(self.hidden_output, self.w_output) + self.b_output
        self.output = self.activation_function(self.output_net)
        return self.output

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.activation_function(self.output_net, derivative=True)
        hidden_error = np.dot(output_delta, self.w_output.T)
        hidden_delta = hidden_error * self.activation_function(self.hidden_net, derivative=True)
        self.w_output += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.b_output += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.w_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        self.b_hidden += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MLP Neural Network")
        
        # Configura a responsividade das colunas e linhas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Frame principal para organizar os widgets
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(sticky="nsew", padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(5, weight=1)

        # Variáveis de controle
        self.train_file = tk.StringVar()
        self.test_file = tk.StringVar()
        self.global_file = tk.StringVar()
        self.hidden_neurons = tk.StringVar(value="5")
        self.learning_rate = tk.StringVar(value="0.1")
        self.max_epochs = tk.StringVar(value="1000")
        self.error_threshold = tk.StringVar(value="0.01")
        self.train_percentage = tk.StringVar(value="80")
        self.activation_function = tk.StringVar(value="logistic")
        self.stop_criterion = tk.StringVar(value="epochs")

        # Área de saída de treinamento
        self.training_output = tk.Text(self.main_frame, height=5, wrap='word')
        self.training_output.grid(row=0, column=0, sticky="nsew", pady=5)

        # Widgets de entrada
        self.create_widgets()
        
        # Tabela para exibir CSV e Resultados
        self.create_table(self.main_frame, row=6)
        self.create_results_table(self.main_frame, row=7)
        
    def create_widgets(self):
        file_frame = ttk.LabelFrame(self.main_frame, text="Arquivos")
        file_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(file_frame, text="Selecionar arquivo de treino", 
                   command=lambda: self.select_train_file()).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(file_frame, text="Selecionar arquivo de teste", 
                   command=lambda: self.select_test_file()).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(file_frame, text="Selecionar arquivo global",
                    command=lambda: self.select_global_file()).grid(row=2, column=0, padx=5, pady=5)
            
        param_frame = ttk.LabelFrame(self.main_frame, text="Parâmetros")
        param_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(param_frame, text="Neurônios na camada oculta:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.hidden_neurons).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(param_frame, text="Porcentagem para treinamento:").grid(row=0, column=3, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.train_percentage).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Taxa de aprendizagem:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.learning_rate).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Função de ativação:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Combobox(param_frame, textvariable=self.activation_function,
                     values=["linear", "logistic", "tanh"]).grid(row=2, column=1, padx=5, pady=5)
        
        stop_frame = ttk.LabelFrame(self.main_frame, text="Critério de Parada")
        stop_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Radiobutton(stop_frame, text="Por épocas", variable=self.stop_criterion,
                        value="epochs").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(stop_frame, textvariable=self.max_epochs).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Radiobutton(stop_frame, text="Por erro", variable=self.stop_criterion,
                        value="error").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(stop_frame, textvariable=self.error_threshold).grid(row=1, column=1, padx=5, pady=5)
        
        # Posicionando o botão "Treinar" na última linha de parâmetros
        ttk.Button(self.main_frame, text="Treinar", command=self.train).grid(row=4, column=0, padx=5, pady=5)
        
    def create_table(self, frame, row):
        # Criação da tabela para exibir o CSV
        self.table_frame = ttk.Frame(frame)
        self.table_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        
        self.table = ttk.Treeview(self.table_frame, show="headings")
        self.table.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    def create_results_table(self, frame, row):
        # Configura a tabela para exibir Época e Erro
        self.results_frame = ttk.Frame(frame)
        self.results_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        
        self.results_table = ttk.Treeview(self.results_frame, columns=("Época", "Erro"), show="headings")
        self.results_table.heading("Época", text="Época")
        self.results_table.heading("Erro", text="Erro")
        self.results_table.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_table.yview)
        self.results_table.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    def select_train_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.global_file.set('')
        self.train_file.set(filename)
        self.display_csv(filename)

    def select_test_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.global_file.set('')
        self.test_file.set(filename)
        self.display_csv(filename)

    def select_global_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.global_file.set(filename)
        self.display_csv(filename)

        # Desabilitar os outros botões quando escolher o arquivo global
        if filename:
            self.train_file.set('')
            self.test_file.set('')
            self.percentage_label.pack(pady=5)
            self.percentage_entry.pack(pady=5)
            # Desabilitar os botões de selecionar arquivo de treino e teste
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Button) and widget.cget("text") == "Escolher arquivo de treino":
                    widget.config(state=tk.DISABLED)
                if isinstance(widget, ttk.Button) and widget.cget("text") == "Escolher arquivo de teste":
                    widget.config(state=tk.DISABLED)  

    def display_csv(self, filename):
        df = pd.read_csv(filename)
        
        # Limpa a tabela antes de inserir novos dados
        self.table.delete(*self.table.get_children())
        self.table["columns"] = list(df.columns)
        for col in df.columns:
            self.table.heading(col, text=col)
            self.table.column(col, anchor="center")

        for row in df.itertuples(index=False):
            self.table.insert("", "end", values=row)     

    def train(self):
        if self.global_file.get():
            self.train_with_single_file()
        elif self.train_file.get() and self.test_file.get():
            self.train_with_two_files()
        else:
            messagebox.showerror("Erro", "Selecione os arquivos corretamente.")
            return

    def train_with_single_file(self):
        if not self.global_file.get():
            tk.messagebox.showerror("Erro", "Por favor, selecione ao menos um arquivo.")
            return

        self.training_output.delete(1.0, tk.END)  # Limpa o texto anterior
        self.training_output.insert(tk.END, "Iniciando o treinamento...\n")
        
        # Carregar dados
        # Pegar o porcentagem de treinamento e teste a partir do train_percentage
        global_data = pd.read_csv(self.global_file.get())
        train_data = global_data.sample(frac=float(self.train_percentage.get()) / 100)
        test_data = global_data.drop(train_data.index)
        
        # Separar features e targets
        X_train = train_data.iloc[:, :-1].values
        y_train = pd.get_dummies(train_data.iloc[:, -1]).values
        X_test = test_data.iloc[:, :-1].values
        y_test = pd.get_dummies(test_data.iloc[:, -1]).values
        
        # Normalizar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Parâmetros da rede
        n_inputs = X_train.shape[1]
        n_hidden = int(self.hidden_neurons.get())
        n_outputs = y_train.shape[1]
        learning_rate = float(self.learning_rate.get())
        
        # Criar a rede neural
        mlp = MLP(n_inputs, n_hidden, n_outputs, learning_rate, self.activation_function.get())
        
        # Configuração do gráfico para erro em tempo real
        fig_error, ax_error = plt.subplots()
        ax_error.set_title("Erro por Época")
        ax_error.set_xlabel("Época")
        ax_error.set_ylabel("Erro (MSE)")
        
        # Lista para armazenar os erros por época
        epoch_errors = []
        line, = ax_error.plot(epoch_errors, label="Erro")

        plt.ion()  # Habilita o modo interativo
        fig_error.show()

        # Treinamento
        max_epochs = int(self.max_epochs.get())
        error_threshold = float(self.error_threshold.get())
        
        for epoch in range(max_epochs):
            # Forward e backward pass
            output = mlp.forward(X_train)
            mlp.backward(X_train, y_train, output)
            
            # Calcular erro (mean squared error)
            loss = np.mean(np.square(y_train - output))
            epoch_errors.append(loss)  # Adiciona o erro à lista
            self.training_output.insert(tk.END, f'Época {epoch + 1}/{max_epochs}, Erro: {loss:.4f}\n')
            self.training_output.see(tk.END)  # Rolagem automática para o final

            # Atualizar o gráfico de erro em tempo real
            line.set_xdata(range(len(epoch_errors)))
            line.set_ydata(epoch_errors)
            ax_error.relim()
            ax_error.autoscale_view()
            fig_error.canvas.draw()
            fig_error.canvas.flush_events()

            # Critério de parada
            if self.stop_criterion.get() == "error" and loss < error_threshold:
                self.training_output.insert(tk.END, "Critério de parada atingido pelo erro.\n")
                break

        # Desabilitar o modo interativo ao final do treinamento
        plt.ioff()

        # Avaliação no conjunto de teste
        test_output = mlp.forward(X_test)
        test_loss = np.mean(np.square(y_test - test_output))
        self.training_output.insert(tk.END, f'Erro no conjunto de teste: {test_loss:.4f}\n')

        # Matriz de Confusão
        predicted_classes = np.argmax(test_output, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        confusion_matrix = pd.crosstab(true_classes, predicted_classes, rownames=['Classe Verdadeira'], colnames=['Classe Predita'])
        self.training_output.insert(tk.END, "Matriz de Confusão:\n")
        self.training_output.insert(tk.END, str(confusion_matrix) + "\n")
        
        # Exibir a matriz de confusão em uma nova janela
        fig_confusion, ax_confusion = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d", ax=ax_confusion)
        ax_confusion.set_xlabel("Classe Predita")
        ax_confusion.set_ylabel("Classe Verdadeira")
        ax_confusion.set_title("Matriz de Confusão")
        fig_confusion.show()

    def train_with_two_files(self):
        if not self.train_file.get():
            tk.messagebox.showerror("Erro", "Por favor, selecione um arquivo de treinamento.")
            return
        if not self.test_file.get():
            tk.messagebox.showerror("Erro", "Por favor, selecione um arquivo de teste.")
            return

        self.training_output.delete(1.0, tk.END)  # Limpa o texto anterior
        self.training_output.insert(tk.END, "Iniciando o treinamento...\n")
        
        # Carregar dados
        train_data = pd.read_csv(self.train_file.get())
        test_data = pd.read_csv(self.test_file.get())
        
        # Separar features e targets
        X_train = train_data.iloc[:, :-1].values
        y_train = pd.get_dummies(train_data.iloc[:, -1]).values
        X_test = test_data.iloc[:, :-1].values
        y_test = pd.get_dummies(test_data.iloc[:, -1]).values
        
        # Normalizar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Parâmetros da rede
        n_inputs = X_train.shape[1]
        n_hidden = int(self.hidden_neurons.get())
        n_outputs = y_train.shape[1]
        learning_rate = float(self.learning_rate.get())
        
        # Criar a rede neural
        mlp = MLP(n_inputs, n_hidden, n_outputs, learning_rate, self.activation_function.get())
        
        # Configuração do gráfico para erro em tempo real
        fig_error, ax_error = plt.subplots()
        ax_error.set_title("Erro por Época")
        ax_error.set_xlabel("Época")
        ax_error.set_ylabel("Erro (MSE)")
        
        # Lista para armazenar os erros por época
        epoch_errors = []
        line, = ax_error.plot(epoch_errors, label="Erro")

        plt.ion()  # Habilita o modo interativo
        fig_error.show()

        # Treinamento
        max_epochs = int(self.max_epochs.get())
        error_threshold = float(self.error_threshold.get())
        
        for epoch in range(max_epochs):
            # Forward e backward pass
            output = mlp.forward(X_train)
            mlp.backward(X_train, y_train, output)
            
            # Calcular erro (mean squared error)
            loss = np.mean(np.square(y_train - output))
            epoch_errors.append(loss)  # Adiciona o erro à lista
            self.training_output.insert(tk.END, f'Época {epoch + 1}/{max_epochs}, Erro: {loss:.4f}\n')
            self.training_output.see(tk.END)  # Rolagem automática para o final

            # Atualizar o gráfico de erro em tempo real
            line.set_xdata(range(len(epoch_errors)))
            line.set_ydata(epoch_errors)
            ax_error.relim()
            ax_error.autoscale_view()
            fig_error.canvas.draw()
            fig_error.canvas.flush_events()

            # Critério de parada
            if self.stop_criterion.get() == "error" and loss < error_threshold:
                self.training_output.insert(tk.END, "Critério de parada atingido pelo erro.\n")
                break

        # Desabilitar o modo interativo ao final do treinamento
        plt.ioff()

        # Avaliação no conjunto de teste
        test_output = mlp.forward(X_test)
        test_loss = np.mean(np.square(y_test - test_output))
        self.training_output.insert(tk.END, f'Erro no conjunto de teste: {test_loss:.4f}\n')

        # Matriz de Confusão
        predicted_classes = np.argmax(test_output, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        confusion_matrix = pd.crosstab(true_classes, predicted_classes, rownames=['Classe Verdadeira'], colnames=['Classe Predita'])
        self.training_output.insert(tk.END, "Matriz de Confusão:\n")
        self.training_output.insert(tk.END, str(confusion_matrix) + "\n")
        
        # Exibir a matriz de confusão em uma nova janela
        fig_confusion, ax_confusion = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d", ax=ax_confusion)
        ax_confusion.set_xlabel("Classe Predita")
        ax_confusion.set_ylabel("Classe Verdadeira")
        ax_confusion.set_title("Matriz de Confusão")
        fig_confusion.show()

if __name__ == "__main__":
    app = Interface()
    app.root.mainloop()