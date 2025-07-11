import sqlite3
import json
import re
import os
import math
import hashlib
import heapq
import difflib
import requests
from datetime import datetime
from uuid import uuid4
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search
from collections import defaultdict
from collections import deque 

# ******************** CONFIGURACIÓN AVANZADA ********************
class AdvancedConfig:
    DB_NAME = "nessy_knowledge_v3.db"
    VECTOR_DB = "nessy_vectors_v3.npy"
    METADATA_DB = "nessy_metadata_v3.json"
    USER_HISTORY_FILE = "user_history_v3.json"
    SIMILARITY_THRESHOLD = 0.65
    CONFIRMATION_THRESHOLD = 2
    MAX_CONTEXT_ATTEMPTS = 3
    MEMORY_DEPTH = 7
    TOP_K_RESULTS = 3
    LEARNING_RATE = 0.3
    DECAY_RATE = 0.98
    MIN_CONFIDENCE = 0.4
    EMBEDDING_SIZE = 256
    MAX_SEARCH_RESULTS = 3
    SEARCH_TIMEOUT = 5
    DEFAULT_DATASET = "default_knowledge.json"

# ******************** MODELO DE EMBEDDINGS MEJORADO ********************
class EnhancedEmbeddingModel:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(max_features=AdvancedConfig.EMBEDDING_SIZE)
        self.trained = False
        self.load_model()

    def train(self, documents):
        if not documents:
            return
            
        # Entrenar modelo TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Construir embeddings promediados
        vocab_size = len(self.vectorizer.get_feature_names_out())
        self.embeddings = np.zeros((len(documents), vocab_size))
        for i, doc in enumerate(documents):
            self.embeddings[i] = tfidf_matrix[i].toarray().mean(axis=0)
        
        # Construir vocabulario
        self.vocab = {word: idx for idx, word in enumerate(self.vectorizer.get_feature_names_out())}
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.trained = True
        self.save_model()

    def embed(self, text):
        if not self.trained:
            return np.zeros(AdvancedConfig.EMBEDDING_SIZE)
            
        words = re.findall(r'\w+', text.lower())
        vector = np.zeros(len(self.vocab))
        
        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] += 1
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
            
        return vector

    def similarity(self, vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return cosine_similarity([vec1], [vec2])[0][0]

    def save_model(self):
        if self.embeddings is not None:
            np.save(AdvancedConfig.VECTOR_DB, self.embeddings)
            with open(AdvancedConfig.METADATA_DB, 'w') as f:
                json.dump({
                    'vocab': self.vocab,
                    'inverse_vocab': self.inverse_vocab,
                    'trained': self.trained
                }, f)

    def load_model(self):
        try:
            self.embeddings = np.load(AdvancedConfig.VECTOR_DB, allow_pickle=True)
            with open(AdvancedConfig.METADATA_DB, 'r') as f:
                data = json.load(f)
                self.vocab = data['vocab']
                self.inverse_vocab = data['inverse_vocab']
                self.trained = data['trained']
        except:
            self.trained = False

# ******************** BUSCADOR DE FUENTES ********************
class SourceFinder:
    @staticmethod
    def find_sources(query, num_results=AdvancedConfig.MAX_SEARCH_RESULTS):
        try:
            return [result for result in search(
                query, 
                num_results=num_results,
                lang='es',
                timeout=AdvancedConfig.SEARCH_TIMEOUT
            )]
        except:
            return []

    @staticmethod
    def fetch_page_content(url):
        try:
            response = requests.get(url, timeout=AdvancedConfig.SEARCH_TIMEOUT)
            if response.status_code == 200:
                return response.text
            return None
        except:
            return None

    @staticmethod
    def extract_relevant_content(content, query, max_length=500):
        if not content:
            return ""
            
        # Buscar fragmentos que contengan palabras clave
        keywords = query.split()
        sentences = re.split(r'(?<=[.!?]) +', content)
        
        relevant = []
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                relevant.append(sentence)
                if len("\n".join(relevant)) > max_length:
                    break
        
        return "\n".join(relevant)

# ******************** BASE DE CONOCIMIENTO MEJORADA ********************
class EnhancedKnowledgeBase:
    def __init__(self, embedding_model):
        self.conn = sqlite3.connect(AdvancedConfig.DB_NAME)
        self.embedding_model = embedding_model
        self._create_tables()
        self._load_knowledge()
        self._load_default_dataset()

    def _create_tables(self):
        cursor = self.conn.cursor()
        # Tabla de conocimiento principal
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                information TEXT NOT NULL,
                source TEXT,
                source_credibility REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                last_updated DATETIME,
                usage_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                embedding BLOB,
                verified INTEGER DEFAULT 0
            )
        ''')
        
        # Tabla de relaciones contextuales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_relations (
                knowledge_id TEXT,
                related_id TEXT,
                strength REAL DEFAULT 1.0,
                last_used DATETIME,
                usage_count INTEGER DEFAULT 0,
                PRIMARY KEY (knowledge_id, related_id),
                FOREIGN KEY (knowledge_id) REFERENCES knowledge (id),
                FOREIGN KEY (related_id) REFERENCES knowledge (id)
            )
        ''')
        
        # Tabla de sesiones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                interaction_count INTEGER DEFAULT 0,
                learning_count INTEGER DEFAULT 0
            )
        ''')
        
        # Tabla de retroalimentación
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id TEXT,
                rating INTEGER,
                comment TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
            )
        ''')
        self.conn.commit()

    def _load_default_dataset(self):
        # Verificar si ya existe conocimiento
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        if cursor.fetchone()[0] > 0:
            return
            
        # Cargar dataset predeterminado si existe
        if os.path.exists(AdvancedConfig.DEFAULT_DATASET):
            with open(AdvancedConfig.DEFAULT_DATASET, 'r') as f:
                try:
                    dataset = json.load(f)
                    for item in dataset:
                        self.add_knowledge(
                            item['topic'],
                            item['information'],
                            source=item.get('source', "Dataset predeterminado")
                        )
                    print(f"✅ Dataset predeterminado cargado: {len(dataset)} conceptos")
                except:
                    print("⚠️ Error cargando dataset predeterminado")

    def _load_knowledge(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT topic, information FROM knowledge")
        documents = [f"{row[0]} {row[1]}" for row in cursor.fetchall()]
        self.embedding_model.train(documents)

    def add_knowledge(self, topic, information, source=None, confidence=1.0, verified=False):
        knowledge_id = str(uuid4())
        embedding = self.embedding_model.embed(f"{topic} {information}")
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO knowledge (id, topic, information, source, confidence, embedding, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (knowledge_id, topic, information, source, confidence, embedding.tobytes(), int(verified)))
        
        self.conn.commit()
        self._load_knowledge()
        return knowledge_id

    def update_knowledge(self, knowledge_id, new_information=None, new_confidence=None, new_source=None):
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if new_information:
            updates.append("information = ?")
            params.append(new_information)
            
            # Actualizar embedding
            cursor.execute("SELECT topic FROM knowledge WHERE id = ?", (knowledge_id,))
            topic = cursor.fetchone()[0]
            embedding = self.embedding_model.embed(f"{topic} {new_information}")
            updates.append("embedding = ?")
            params.append(embedding.tobytes())
        
        if new_confidence is not None:
            updates.append("confidence = ?")
            params.append(new_confidence)
            
        if new_source:
            updates.append("source = ?")
            params.append(new_source)
            
        if updates:
            updates.append("last_updated = CURRENT_TIMESTAMP")
            query = f"UPDATE knowledge SET {', '.join(updates)} WHERE id = ?"
            params.append(knowledge_id)
            cursor.execute(query, params)
            self.conn.commit()
            self._load_knowledge()

    def record_feedback(self, knowledge_id, rating, comment=None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (knowledge_id, rating, comment)
            VALUES (?, ?, ?)
        ''', (knowledge_id, rating, comment))
        self.conn.commit()
        
        # Actualizar confianza basada en feedback
        cursor.execute("SELECT confidence FROM knowledge WHERE id = ?", (knowledge_id,))
        row = cursor.fetchone()
        if not row:
            return
            
        current_confidence = row[0]
        
        # Ajustar confianza: 5 estrellas aumenta, 1 estrella disminuye
        adjustment = 0.1 * (rating - 3)  # Ajuste basado en desviación de 3 estrellas
        new_confidence = max(AdvancedConfig.MIN_CONFIDENCE, min(1.0, current_confidence + adjustment))
        self.update_knowledge(knowledge_id, new_confidence=new_confidence)

    def find_similar_knowledge(self, query, threshold=AdvancedConfig.SIMILARITY_THRESHOLD):
        query_embedding = self.embedding_model.embed(query)
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, topic, information, confidence, embedding, verified FROM knowledge")
        results = []
        
        for kid, topic, info, confidence, emb_blob, verified in cursor.fetchall():
            try:
                embedding = np.frombuffer(emb_blob, dtype=np.float64)
            except:
                continue
                
            similarity = self.embedding_model.similarity(query_embedding, embedding)
            
            # Ajustar similitud por confianza y verificación
            adjusted_similarity = similarity * confidence
            if verified:
                adjusted_similarity *= 1.2  # Boost a conocimiento verificado
            
            if adjusted_similarity >= threshold:
                results.append({
                    'id': kid,
                    'topic': topic,
                    'information': info,
                    'similarity': similarity,
                    'confidence': confidence,
                    'adjusted_similarity': adjusted_similarity,
                    'verified': bool(verified)
                })
        
        return heapq.nlargest(
            AdvancedConfig.TOP_K_RESULTS, 
            results, 
            key=lambda x: x['adjusted_similarity']
        )

    def get_knowledge(self, knowledge_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM knowledge WHERE id=?", (knowledge_id,))
        return cursor.fetchone()

    def update_usage(self, knowledge_id):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE knowledge 
            SET last_used=CURRENT_TIMESTAMP, usage_count=usage_count+1 
            WHERE id=?
        ''', (knowledge_id,))
        self.conn.commit()

    def add_context_relation(self, knowledge_id, related_id):
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO context_relations (knowledge_id, related_id, last_used)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(knowledge_id, related_id) DO UPDATE SET
                strength = strength + 0.1,
                usage_count = usage_count + 1,
                last_used = CURRENT_TIMESTAMP
            ''', (knowledge_id, related_id))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def start_session(self):
        session_id = str(uuid4())
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id) 
            VALUES (?)
        ''', (session_id,))
        self.conn.commit()
        return session_id

    def end_session(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET end_time=CURRENT_TIMESTAMP 
            WHERE session_id=?
        ''', (session_id,))
        self.conn.commit()

    def record_interaction(self, session_id, learned=False):
        cursor = self.conn.cursor()
        if learned:
            cursor.execute('''
                UPDATE sessions 
                SET interaction_count=interaction_count+1, 
                    learning_count=learning_count+1 
                WHERE session_id=?
            ''', (session_id,))
        else:
            cursor.execute('''
                UPDATE sessions 
                SET interaction_count=interaction_count+1 
                WHERE session_id=?
            ''', (session_id,))
        self.conn.commit()

# ******************** SISTEMA DE APRENDIZAJE AVANZADO ********************
class AdvancedLearningSystem:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.confirmation_count = defaultdict(int)
        self.confidence_decay = AdvancedConfig.DECAY_RATE

    def learn_from_user(self, topic, session_id):
        print(f"\n🔍 Nessy: Mi conocimiento sobre '{topic}' es limitado. ¿Podrías enseñarme o debo investigar?")
        print("1. Enseñarme ahora")
        print("2. Buscar en fuentes confiables")
        
        try:
            choice = int(input("Tu elección (1-2): "))
        except:
            choice = 1
            
        if choice == 2:
            return self._learn_from_external(topic, session_id)
        return self._learn_from_user(topic, session_id)

    def _learn_from_user(self, topic, session_id):
        information = self._gather_information(topic)
        
        if information:
            source = input("\n🔗 ¿Tienes alguna fuente confiable para esto? (opcional): ").strip()
            confidence = self._assess_confidence(information, source)
            knowledge_id = self.kb.add_knowledge(topic, information, source, confidence)
            
            print(f"\n✅ Nessy: ¡Gracias! He aprendido sobre '{topic}' con confianza {confidence:.2f}/1.0")
            self.kb.record_interaction(session_id, learned=True)
            return knowledge_id
        return None

    def _learn_from_external(self, topic, session_id):
        print(f"\n🔎 Nessy: Buscando información confiable sobre '{topic}'...")
        sources = SourceFinder.find_sources(topic)
        
        if not sources:
            print("⚠️ No encontré fuentes confiables. ¿Podrías enseñarme tú?")
            return self._learn_from_user(topic, session_id)
            
        print("\n📚 Fuentes encontradas:")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source}")
            
        try:
            selection = int(input("\nSelecciona una fuente para verificar (0 para cancelar): "))
            if selection == 0:
                return self._learn_from_user(topic, session_id)
                
            selected_source = sources[selection-1]
            content = SourceFinder.fetch_page_content(selected_source)
            if not content:
                print("⚠️ No pude acceder al contenido. Intentemos con otra fuente.")
                return self._learn_from_external(topic, session_id)
                
            information = SourceFinder.extract_relevant_content(content, topic)
            
            if not information:
                print("⚠️ No encontré información relevante. ¿Podrías enseñarme tú?")
                return self._learn_from_user(topic, session_id)
                
            print(f"\n📝 Información encontrada:\n{information[:500]}...")
            
            # Solicitar confirmación del usuario
            confirm = input("\n¿Es esta información correcta? (sí/no): ").lower() in ['sí', 'si', 's', 'yes', 'y']
            if confirm:
                confidence = 0.9  # Alta confianza por verificación externa
                knowledge_id = self.kb.add_knowledge(topic, information, selected_source, confidence, verified=True)
                print(f"\n✅ Nessy: ¡Gracias! He aprendido sobre '{topic}' desde una fuente confiable")
                self.kb.record_interaction(session_id, learned=True)
                return knowledge_id
                
            print("⚠️ Entendido, descartaré esta información.")
            return self._learn_from_external(topic, session_id)
            
        except:
            print("⚠️ Selección inválida, intentemos nuevamente.")
            return self._learn_from_external(topic, session_id)

    def _gather_information(self, topic):
        information = ""
        for i in range(AdvancedConfig.MAX_CONTEXT_ATTEMPTS):
            info_input = input(f"\n💡 [{i+1}/{AdvancedConfig.MAX_CONTEXT_ATTEMPTS}] Información sobre '{topic}': ").strip()
            
            if not info_input:
                if i == 0:
                    print("⚠️ Debes proporcionar al menos algo de información")
                    continue
                break
                
            information += info_input + " "
            
            # Verificar si el usuario quiere añadir más
            if i < AdvancedConfig.MAX_CONTEXT_ATTEMPTS - 1:
                more = input("¿Añadir más detalles? (s/n): ").strip().lower()
                if more not in ['s', 'si', 'y', 'yes']:
                    break
        
        return information.strip()

    def _assess_confidence(self, information, source):
        confidence = 1.0
        
        # Factores que reducen confianza
        if len(information) < 20:
            confidence *= 0.7  # Información muy corta
        if not source:
            confidence *= 0.8  # Falta de fuente
        if 'creo que' in information.lower() or 'probablemente' in information.lower():
            confidence *= 0.6  # Lenguaje incierto
            
        return max(AdvancedConfig.MIN_CONFIDENCE, confidence)

    def request_confirmation(self, knowledge_id, topic, information):
        self.confirmation_count[knowledge_id] += 1
        
        print(f"\n🧠 Nessy: Por favor confirma mi comprensión de '{topic}':")
        print(f"📝 {information}")
        
        if self.confirmation_count[knowledge_id] >= AdvancedConfig.CONFIRMATION_THRESHOLD:
            print("⚠️ He pedido confirmación varias veces sobre este tema")
        
        response = input("\n¿Es correcto? (sí/no/parcialmente): ").strip().lower()
        
        if response in ['sí', 'si', 's', 'yes', 'y']:
            return True, None
        elif response in ['parcialmente', 'parcial', 'p']:
            correction = input("¿Qué parte necesita corrección? ")
            return False, correction
        else:
            return False, None

    def decay_confidence(self, knowledge_id):
        cursor = self.kb.conn.cursor()
        cursor.execute("SELECT confidence FROM knowledge WHERE id=?", (knowledge_id,))
        row = cursor.fetchone()
        if not row:
            return 0.0
            
        current_confidence = row[0]
        new_confidence = max(AdvancedConfig.MIN_CONFIDENCE, current_confidence * self.confidence_decay)
        self.kb.update_knowledge(knowledge_id, new_confidence=new_confidence)
        return new_confidence

    def process_feedback(self, knowledge_id):
        print("\n🌟 Nessy: ¿Cómo de útil fue esta información?")
        print("1: Poco útil ⭐")
        print("2: Útil ⭐⭐")
        print("3: Muy útil ⭐⭐⭐")
        print("4: Excelente ⭐⭐⭐⭐")
        print("5: Excepcional ⭐⭐⭐⭐⭐")
        
        try:
            rating = int(input("Calificación (1-5): "))
            if 1 <= rating <= 5:
                comment = input("Comentario (opcional): ")
                self.kb.record_feedback(knowledge_id, rating, comment)
                print("✅ ¡Gracias por tu feedback!")
                return True
            else:
                print("⚠️ Calificación inválida")
        except:
            print("⚠️ Debes ingresar un número entre 1 y 5")
        return False

# ******************** MEMORIA CONTEXTUAL AVANZADA ********************
class AdvancedContextMemory:
    def __init__(self, depth=AdvancedConfig.MEMORY_DEPTH):
        self.memory = deque(maxlen=depth)
        self.current_context = {}
    
    def add_context(self, user_input, response, knowledge_id=None):
        context = {
            'user': user_input,
            'response': response,
            'knowledge_id': knowledge_id,
            'timestamp': datetime.now()
        }
        self.memory.append(context)
        self.current_context = context
    
    def get_recent_context(self):
        return list(self.memory)
    
    def get_relevant_memories(self, query, embedding_model):
        query_embed = embedding_model.embed(query)
        memories = []
        
        for memory in self.memory:
            text = f"{memory['user']} {memory['response']}"
            memory_embed = embedding_model.embed(text)
            similarity = embedding_model.similarity(query_embed, memory_embed)
            memories.append((similarity, memory))
        
        return [mem for _, mem in sorted(memories, key=lambda x: x[0], reverse=True)]

# ******************** GENERADOR DE RESPUESTAS INTELIGENTE ********************
class SmartResponseGenerator:
    def __init__(self, knowledge_base, learning_system, context_memory):
        self.kb = knowledge_base
        self.ls = learning_system
        self.memory = context_memory
    
    def generate_response(self, user_input, session_id):
        # Primero verificar si es un comando especial
        if user_input.startswith('/'):
            return None, "Comando procesado"
        
        # Buscar conocimiento existente
        matches = self.kb.find_similar_knowledge(user_input)
        
        if matches:
            best_match = matches[0]
            self.kb.update_usage(best_match['id'])
            
            # Verificar si necesitamos confirmación
            if best_match['confidence'] < 0.7 and not best_match['verified']:
                confirm, correction = self.ls.request_confirmation(
                    best_match['id'], 
                    best_match['topic'], 
                    best_match['information']
                )
                
                if not confirm:
                    if correction:
                        # Aplicar corrección parcial
                        self.kb.update_knowledge(
                            best_match['id'], 
                            new_information=correction
                        )
                        response = f"\n🔄 Nessy: He actualizado mi conocimiento sobre '{best_match['topic']}'"
                    else:
                        # Disminuir confianza
                        new_conf = self.ls.decay_confidence(best_match['id'])
                        response = f"\n⚠️ Nessy: He reducido mi confianza sobre este tema a {new_conf:.2f}/1.0"
                    self.memory.add_context(user_input, response, best_match['id'])
                    return best_match['id'], response
            
            # Construir respuesta basada en contexto
            response = self._build_contextual_response(best_match, user_input)
            
            # Añadir relaciones contextuales
            if self.memory.current_context and self.memory.current_context.get('knowledge_id'):
                self.kb.add_context_relation(
                    self.memory.current_context['knowledge_id'],
                    best_match['id']
                )
            
            # Ocasionalmente solicitar feedback
            if np.random.random() < 0.2:  # 20% de probabilidad
                response += "\n\n¿Te resultó útil esta información? (responde con 'feedback' para calificar)"
            
            self.memory.add_context(user_input, response, best_match['id'])
            return best_match['id'], response
        
        # Aprendizaje de nuevo conocimiento
        knowledge_id = self.ls.learn_from_user(user_input, session_id)
        if knowledge_id:
            response = f"\n🧠 Nessy: Ahora sé sobre '{user_input}'. ¿En qué más puedo ayudarte?"
            self.memory.add_context(user_input, response, knowledge_id)
            return knowledge_id, response
        
        response = "\n❓ Nessy: Tal vez podamos hablar de otro tema. ¿Qué te gustaría aprender o discutir?"
        self.memory.add_context(user_input, response)
        return None, response

    def _build_contextual_response(self, knowledge, user_input):
        response = f"\n🧠 Nessy: Sobre '{knowledge['topic']}':\n"
        response += f"📚 {knowledge['information']}"
        
        # Añadir fuente si está disponible
        if source := self._get_knowledge_source(knowledge['id']):
            response += f"\n🔍 Fuente: {source}"
        
        # Añadir contexto de memoria si es relevante
        relevant_memories = self.memory.get_relevant_memories(
            knowledge['topic'], 
            self.kb.embedding_model
        )[:1]  # Memoria más relevante
        
        if relevant_memories:
            mem = relevant_memories[0]
            response += f"\n\n💭 Recuerdo que antes hablamos de: {mem['user'][:50]}..."
        
        # Añadir sugerencia de profundización
        response += self._generate_follow_up(knowledge)
        return response

    def _get_knowledge_source(self, knowledge_id):
        knowledge = self.kb.get_knowledge(knowledge_id)
        return knowledge[3] if knowledge and knowledge[3] else None

    def _generate_follow_up(self, knowledge):
        follow_ups = [
            "\n\n¿Te gustaría profundizar en algún aspecto específico?",
            "\n\n¿Quieres que relacione esto con algún otro tema?",
            "\n\n¿Deseas explorar aplicaciones prácticas de esto?",
            "\n\n¿Te interesa ver fuentes adicionales?"
        ]
        return follow_ups[hash(knowledge['id']) % len(follow_ups)]

# ******************** SISTEMA DE ENTRENAMIENTO AVANZADO ********************
class AdvancedTrainingSystem:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def train_from_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Procesar diferentes formatos
            if filename.endswith('.json'):
                return self._train_from_json(content, filename)
            else:
                return False, "Formato de archivo no soportado"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _train_from_json(self, content, filename):
        try:
            data = json.loads(content)
            count = 0
            
            if isinstance(data, list):
                for item in data:
                    if 'topic' in item and 'information' in item:
                        source = item.get('source', f"Archivo: {os.path.basename(filename)}")
                        self.kb.add_knowledge(
                            item['topic'], 
                            item['information'], 
                            source,
                            verified=True
                        )
                        count += 1
            elif isinstance(data, dict):
                for topic, info in data.items():
                    self.kb.add_knowledge(
                        topic, 
                        info, 
                        source=f"Archivo: {os.path.basename(filename)}",
                        verified=True
                    )
                    count += 1
            
            return True, f"Entrenado con {count} conceptos de {filename}"
        except json.JSONDecodeError:
            return False, "Error decodificando JSON"

# ******************** INTERFAZ DE USUARIO COMPLETA ********************
class NessyInterface:
    def __init__(self):
        self.embedding_model = EnhancedEmbeddingModel()
        self.kb = EnhancedKnowledgeBase(self.embedding_model)
        self.ls = AdvancedLearningSystem(self.kb)
        self.memory = AdvancedContextMemory()
        self.rg = SmartResponseGenerator(self.kb, self.ls, self.memory)
        self.trainer = AdvancedTrainingSystem(self.kb)
        self.session_id = self.kb.start_session()
        self.user_id = self._get_user_id()
    
    def _get_user_id(self):
        try:
            with open('nessy_user_id.txt', 'r') as f:
                return f.read().strip()
        except:
            user_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:12]
            with open('nessy_user_id.txt', 'w') as f:
                f.write(user_id)
            return user_id
    
    def display_welcome(self):
        print("\n" + "="*70)
        print(f"🌀 NESSY - Sistema de Inteligencia Avanzada (v3.0)".center(70))
        print("="*70)
        print("\n🔍 Soy un sistema de IA generativa con aprendizaje profundo")
        print("📚 Mi conocimiento evoluciona con cada interacción")
        print("💡 Puedes entrenarme con datos, corregirme y evaluar mis respuestas")
        print("🔍 Busco automáticamente en fuentes confiables cuando es necesario")
        print("📂 Comandos especiales: /entrenar, /retroalimentacion, /memoria, /estado")
        print("✋ Escribe 'salir' para terminar la sesión\n")
        print(f"🆔 Sesión: {self.session_id} | 👤 Usuario: {self.user_id}")
        print("-"*70)
    
    def run(self):
        self.display_welcome()
        print("\n💬 Comencemos nuestra conversación. ¿Qué te gustaría discutir o aprender hoy?")
        
        try:
            while True:
                user_input = input("\n👤 Tú: ").strip()
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break
                
                # Manejar comandos especiales
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # Manejar feedback explícito
                if user_input.lower() == 'feedback' and self.memory.current_context.get('knowledge_id'):
                    self.ls.process_feedback(self.memory.current_context['knowledge_id'])
                    continue
                
                self.kb.record_interaction(self.session_id)
                
                # Generar y mostrar respuesta
                knowledge_id, response = self.rg.generate_response(user_input, self.session_id)
                print(response)
        
        finally:
            self.kb.end_session(self.session_id)
            self.display_session_summary()
    
    def handle_command(self, command):
        parts = command.split()
        cmd = parts[0][1:].lower()
        
        if cmd == 'entrenar' and len(parts) > 1:
            filename = parts[1]
            success, message = self.trainer.train_from_file(filename)
            print(f"\n🔮 Nessy: {message}")
        
        elif cmd == 'retroalimentacion' and self.memory.current_context.get('knowledge_id'):
            self.ls.process_feedback(self.memory.current_context['knowledge_id'])
        
        elif cmd == 'memoria':
            print("\n💾 Memoria contextual reciente:")
            memories = self.memory.get_recent_context()
            for i, mem in enumerate(memories, 1):
                print(f"{i}. Tú: {mem['user'][:50]}... → Nessy: {mem['response'][:50]}...")
        
        elif cmd == 'estado':
            cursor = self.kb.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            knowledge_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM feedback")
            feedback_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM knowledge WHERE verified = 1")
            verified_count = cursor.fetchone()[0]
            
            print("\n📊 Estado del sistema:")
            print(f"- Conceptos aprendidos: {knowledge_count}")
            print(f"- Conceptos verificados: {verified_count}")
            print(f"- Retroalimentaciones recibidas: {feedback_count}")
            print(f"- Profundidad de memoria: {len(self.memory.memory)}/{AdvancedConfig.MEMORY_DEPTH}")
        
        else:
            print("\n⚠️ Comando no reconocido. Comandos disponibles:")
            print("/entrenar [archivo] - Entrena con un archivo JSON")
            print("/retroalimentacion - Califica la última respuesta")
            print("/memoria - Muestra el contexto reciente")
            print("/estado - Muestra estadísticas del sistema")
    
    def display_session_summary(self):
        cursor = self.kb.conn.cursor()
        cursor.execute('''
            SELECT interaction_count, learning_count 
            FROM sessions 
            WHERE session_id=?
        ''', (self.session_id,))
        row = cursor.fetchone()
        interactions = row[0] if row else 0
        learnings = row[1] if row else 0
        
        print("\n" + "="*70)
        print("📊 Resumen de sesión".center(70))
        print("="*70)
        print(f"🕒 Duración: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💬 Interacciones: {interactions}")
        print(f"🎓 Nuevos aprendizajes: {learnings}")
        print(f"🧠 Conceptos en memoria: {len(self.memory.memory)}")
        print("="*70)
        print("Gracias por entrenar a Nessy. ¡Hasta la próxima!".center(70))
        print("="*70)

# ******************** EJECUCIÓN PRINCIPAL ********************
if __name__ == "__main__":
    # Crear dataset predeterminado si no existe
    if not os.path.exists(AdvancedConfig.DEFAULT_DATASET):
        default_knowledge = [
            {
                "topic": "Inteligencia Artificial",
                "information": "La inteligencia artificial es el campo de estudio que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
                "source": "Fundamentos de IA, Russell & Norvig"
            },
            {
                "topic": "Machine Learning",
                "information": "El aprendizaje automático es una rama de la IA que se centra en desarrollar algoritmos que permiten a las computadoras aprender de los datos.",
                "source": "Pattern Recognition and Machine Learning, Bishop"
            },
            {
                "topic": "Redes Neuronales",
                "information": "Las redes neuronales artificiales son modelos computacionales inspirados en el cerebro humano que consisten en capas de nodos interconectados.",
                "source": "Deep Learning, Goodfellow et al."
            }
        ]
        with open(AdvancedConfig.DEFAULT_DATASET, 'w') as f:
            json.dump(default_knowledge, f, indent=2)
    
    try:
        interface = NessyInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Sesión interrumpida. Guardando progreso...")
        interface.kb.end_session(interface.session_id)
        interface.display_session_summary()
    except Exception as e:
        print(f"\n⚠️ Error crítico: {str(e)}")
        print("Por favor reporta este error con el ID de sesión")