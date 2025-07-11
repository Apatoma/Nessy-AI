import sqlite3
import json
import difflib
import re
from datetime import datetime
from uuid import uuid4
import hashlib
import os

# Configuración inicial
class Config:
    DB_NAME = "nessy_knowledge.db"
    USER_HISTORY_FILE = "user_history.json"
    SIMILARITY_THRESHOLD = 0.6
    CONFIRMATION_THRESHOLD = 2
    MAX_CONTEXT_ATTEMPTS = 3

# Sistema de base de conocimiento
class KnowledgeBase:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_NAME)
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        # Tabla de conocimiento
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                information TEXT NOT NULL,
                source TEXT,
                source_credibility INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        # Tabla de relaciones contextuales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_relations (
                knowledge_id TEXT,
                related_id TEXT,
                strength REAL DEFAULT 1.0,
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
                interaction_count INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()
    
    def add_knowledge(self, topic, information, source=None):
        knowledge_id = str(uuid4())
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO knowledge (id, topic, information, source)
            VALUES (?, ?, ?, ?)
        ''', (knowledge_id, topic, information, source))
        
        self.conn.commit()
        return knowledge_id
    
    def find_similar_knowledge(self, topic, threshold=Config.SIMILARITY_THRESHOLD):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, topic, information FROM knowledge")
        all_knowledge = cursor.fetchall()
        
        matches = []
        for kid, ktopic, kinfo in all_knowledge:
            topic_sim = difflib.SequenceMatcher(None, topic.lower(), ktopic.lower()).ratio()
            info_sim = difflib.SequenceMatcher(None, topic.lower(), kinfo.lower()).ratio()
            similarity = max(topic_sim, info_sim)
            
            if similarity >= threshold:
                matches.append({
                    'id': kid,
                    'topic': ktopic,
                    'information': kinfo,
                    'similarity': similarity
                })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
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
                INSERT INTO context_relations (knowledge_id, related_id)
                VALUES (?, ?)
            ''', (knowledge_id, related_id))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass
    
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
    
    def record_interaction(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET interaction_count=interaction_count+1 
            WHERE session_id=?
        ''', (session_id,))
        self.conn.commit()

# Módulo de aprendizaje interactivo
class LearningModule:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.learning_context = {}
    
    def learn_from_user(self, topic):
        print(f"\n🔍 Nessy: No tengo información sobre '{topic}'. ¿Podrías enseñarme?")
        attempts = 0
        
        while attempts < Config.MAX_CONTEXT_ATTEMPTS:
            information = input("💡 Por favor, proporciona información sobre este tema: ").strip()
            source = input("🔗 ¿Tienes alguna fuente para verificar? (opcional): ").strip()
            
            if information:
                confirmation = self._request_confirmation(topic, information, source)
                if confirmation:
                    knowledge_id = self.kb.add_knowledge(topic, information, source)
                    print(f"\n✅ Nessy: ¡Gracias! He aprendido sobre '{topic}' y lo he almacenado.")
                    return knowledge_id
                else:
                    print("\n🔄 Nessy: Entendido, intentemos nuevamente.")
            
            attempts += 1
        
        print("\n⏭ Nessy: Continuemos con otro tema. Puedes volver a este más tarde.")
        return None
    
    def _request_confirmation(self, topic, information, source):
        print(f"\n🧠 Nessy: Por favor confirma la información:")
        print(f"📌 Tema: {topic}")
        print(f"📝 Información: {information}")
        if source:
            print(f"🔍 Fuente: {source}")
        
        response = input("\n¿Es correcta esta información? (sí/no): ").strip().lower()
        return response in ['sí', 'si', 's', 'yes', 'y']
    
    def enhance_context(self, knowledge_id, user_input):
        related_knowledge = self.kb.find_similar_knowledge(user_input, threshold=0.4)
        for item in related_knowledge:
            if item['id'] != knowledge_id:
                self.kb.add_context_relation(knowledge_id, item['id'])

# Generador de respuestas
class ResponseGenerator:
    def __init__(self, knowledge_base, learning_module):
        self.kb = knowledge_base
        self.lm = learning_module
        self.context_history = []
    
    def generate_response(self, user_input):
        self.context_history.append(user_input[:50])
        if len(self.context_history) > 5:
            self.context_history.pop(0)
        
        # Buscar conocimiento existente
        matches = self.kb.find_similar_knowledge(user_input)
        
        if matches:
            best_match = matches[0]
            self.kb.update_usage(best_match['id'])
            self.lm.enhance_context(best_match['id'], user_input)
            
            response = f"\n🧠 Nessy: Sobre '{best_match['topic']}':\n"
            response += f"📚 {best_match['information']}"
            
            if source := self._get_knowledge_source(best_match['id']):
                response += f"\n🔍 Fuente: {source}"
            
            response += self._generate_follow_up(best_match)
            return response
        
        # Aprendizaje de nuevo conocimiento
        knowledge_id = self.lm.learn_from_user(user_input)
        if knowledge_id:
            return f"\n🧠 Nessy: Ahora sé sobre '{user_input}'. ¿En qué más puedo ayudarte?"
        
        return "\n❓ Nessy: Tal vez podamos hablar de otro tema. ¿Qué más te interesa?"

    def _get_knowledge_source(self, knowledge_id):
        knowledge = self.kb.get_knowledge(knowledge_id)
        return knowledge[3] if knowledge and knowledge[3] else None

    def _generate_follow_up(self, knowledge):
        follow_ups = [
            "\n\n¿Te gustaría saber más sobre algún aspecto específico?",
            "\n\n¿Quieres que profundice en algún punto relacionado?",
            "\n\n¿Te interesa explorar aplicaciones prácticas de esto?",
            "\n\n¿Deseas contrastar esto con otra perspectiva?"
        ]
        return follow_ups[len(knowledge['id']) % len(follow_ups)]

# Gestor de historial de usuario
class UserHistoryManager:
    def __init__(self):
        self.history_file = Config.USER_HISTORY_FILE
        self.history = self._load_history()
    
    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_interaction(self, session_id, user_input, response):
        session_history = self.history.get(session_id, [])
        session_history.append({
            'timestamp': str(datetime.now()),
            'user': user_input,
            'nessy': response
        })
        self.history[session_id] = session_history
        self.save_history()

# Interfaz de usuario en terminal
class TerminalInterface:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.lm = LearningModule(self.kb)
        self.rg = ResponseGenerator(self.kb, self.lm)
        self.uhm = UserHistoryManager()
        self.session_id = self.kb.start_session()
        self.user_id = self._get_user_id()
    
    def _get_user_id(self):
        try:
            with open('user_id.txt', 'r') as f:
                return f.read().strip()
        except:
            user_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]
            with open('user_id.txt', 'w') as f:
                f.write(user_id)
            return user_id
    
    def display_welcome(self):
        print("\n" + "="*60)
        print(f"🌟 Bienvenido a NESSY - Asistente Cognitivo Avanzado".center(60))
        print("="*60)
        print("\n🔍 Soy un sistema de IA generativa que aprende interactivamente")
        print("📚 Mi conocimiento evoluciona con cada interacción")
        print("💡 Puedes enseñarme nuevos conceptos y verificar mis fuentes")
        print("✋ Escribe 'salir' en cualquier momento para terminar la sesión\n")
        print(f"🆔 ID de Sesión: {self.session_id}")
        print(f"👤 ID de Usuario: {self.user_id}")
        print("-"*60 + "\n")
    
    def run(self):
        self.display_welcome()
        
        try:
            while True:
                user_input = input("\n👤 Tú: ").strip()
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break
                
                self.kb.record_interaction(self.session_id)
                
                # Generar y mostrar respuesta
                response = self.rg.generate_response(user_input)
                print(response)
                
                # Guardar en historial
                self.uhm.add_interaction(self.session_id, user_input, response)
        
        finally:
            self.kb.end_session(self.session_id)
            print("\n" + "="*60)
            print("💾 Sesión guardada. Gracias por interactuar con NESSY".center(60))
            print(f"📚 Total de interacciones: {self._get_interaction_count()}")
            print("="*60)
    
    def _get_interaction_count(self):
        cursor = self.kb.conn.cursor()
        cursor.execute("SELECT interaction_count FROM sessions WHERE session_id=?", (self.session_id,))
        return cursor.fetchone()[0]

# Punto de entrada principal
if __name__ == "__main__":
    try:
        interface = TerminalInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Sesión interrumpida. Guardando progreso...")
    except Exception as e:
        print(f"\n⚠️ Error crítico: {str(e)}")
        print("Por favor reporta este error con el ID de sesión")