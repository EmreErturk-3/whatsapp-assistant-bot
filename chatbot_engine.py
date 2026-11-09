import re
import json
from typing import Dict, List, Tuple
import ollama
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedTurkishChatbotEngine:
    """Advanced Turkish chatbot using semantic search for superior accuracy."""

    def __init__(self, knowledge_base_path: str, model_name: str = "gemma2:2b", embedding_model: str = "all-minilm"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_chunks = []
        self.chunk_embeddings = []
        self.conversation_history = []

        logger.info("Chatbot Engine başlatılıyor")
        self._load_knowledge_base()
        self._test_ollama_connection()
        self._generate_embeddings()

    def _load_knowledge_base(self):
        """Load and process knowledge base."""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into meaningful chunks
            sections = self._smart_split_content(content)
            self.knowledge_chunks = sections

            logger.info(f"📚 {len(self.knowledge_chunks)} bilgi parçası yüklendi")

        except FileNotFoundError:
            logger.error(f"❌ Bilgi tabanı bulunamadı: {self.knowledge_base_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Bilgi tabanı yükleme hatası: {e}")
            raise

    def _smart_split_content(self, content: str) -> List[str]:
        """Intelligently split content into meaningful chunks."""
        chunks = []
        sections = re.split(r'\n(?=#{1,3}\s)', content)

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) > 1000:
                paragraphs = section.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para and len(para) > 50:
                        chunks.append(para)
            else:
                if len(section) > 50:
                    chunks.append(section)
        return chunks

    def _test_ollama_connection(self):
        """Test if Ollama is available."""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt="Test",
                options={'num_predict': 5}
            )
            logger.info("✅ Ollama bağlantısı başarılı")
        except Exception as e:
            logger.error(f"❌ Ollama bağlantı hatası: {e}")
            logger.error("💡 'ollama serve' komutu ile Ollama'yı başlatın")
            raise

    def _generate_embeddings(self):
        """Generate and store embeddings for all knowledge chunks."""
        logger.info(f"🧠 {len(self.knowledge_chunks)} parça için embedding oluşturuluyor...")

        try:
            # Check if embedding model is available
            _ = ollama.embeddings(model=self.embedding_model, prompt='test')
        except Exception:
            logger.warning(f"Embedding modeli '{self.embedding_model}' bulunamadı. "
                           f"Ollama'dan '{self.embedding_model}' modelini indirin.")
            raise

        self.chunk_embeddings = [
            ollama.embeddings(model=self.embedding_model, prompt=chunk)['embedding']
            for chunk in self.knowledge_chunks
        ]
        logger.info("✅ Embedding'ler başarıyla oluşturuldu")

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5
        if not magnitude_a or not magnitude_b:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)

    def find_relevant_content(self, question: str, max_chunks: int = 3) -> List[Tuple[str, float]]:
        """Find most relevant content chunks using semantic search."""
        if not self.knowledge_chunks or not self.chunk_embeddings:
            return []

        # Generate embedding for the user's question
        question_embedding = ollama.embeddings(
            model=self.embedding_model, prompt=question
        )['embedding']

        # Calculate similarity scores
        chunk_scores = []
        for i, chunk in enumerate(self.knowledge_chunks):
            similarity = self._cosine_similarity(question_embedding, self.chunk_embeddings[i])
            chunk_scores.append((chunk, similarity))

        # Sort by score and return top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter out chunks with very low relevance
        top_chunks = [c for c in chunk_scores if c[1] > 0.4]

        logger.info(f"🔍 {len(top_chunks)} ilgili parça bulundu.")
        if top_chunks:
            logger.info(f"🏆 En yüksek benzerlik skoru: {top_chunks[0][1]:.2f}")

        return top_chunks[:max_chunks]

    def _create_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Create optimized prompt for Turkish responses."""
        # Special handling for basic greetings - don't use the fallback prompt
        if question.strip().lower() == 'merhaba':
            return f"""
               Ben TechSoft şirketinin müşteri hizmetleri asistanıyım.

               MÜŞTERİ MESAJI: {question}

               Müşteriyi nazikçe karşıla ve yardım teklif et. Kısa ve samimi yanıt ver. 

               YANIT:"""
        if not context_chunks:
            # Improved prompt for no-context
            return f"""
            Ben TechSoft şirketinin nazik ve yardımsever müşteri hizmetleri asistanıyım müşteriye destek olmak için varım.

            MÜŞTERİ SORUSU: {question}

            Bu soru hakkında bilgi tabanımda detaylı bilgi bulamadım. Lütfen şu bilgiyi kullanarak kısa, nazik ve yardımsever bir Türkçe yanıt ver:
            "Üzgünüm, bu konu hakkında elimde yeterli bilgi bulunmuyor. Daha fazla detay için lütfen bize info@techsoft.com.tr adresinden e-posta gönderin veya +90 212 555 0123 numaralı telefondan ulaşın. Size yardımcı olmaktan memnuniyet duyarız."
            """

        context = "\n\n---\n\n".join(context_chunks)

        prompt = f"""
        
        Müşteri sorusunu yanıtlamak için yalnızca aşağıda verilen şirket bilgilerini kullan.

        ŞİRKET BİLGİ TABANI:
        {context}

        MÜŞTERİ SORUSU: {question}

        YANIT KURALLARI:
        1. Tamamen Türkçe yanıt ver.
        2. Profesyonel Call center çalışanı gibi bir dil kullan.
        3. Maksimum 2 cümle ile yanıtla.
        4. Bilgi tabanındaki spesifik verileri doğrudan kullan.
        5. Kesin bilgi yoksa veya bilgi yetersizse, müşteriyi iletişim bilgilerine yönlendirerek "Yardımcı olmaktan memnuniyet duyarız." gibi nazik bir ifade ekle.
        6. kısa ve öz yanıt ver net ol 
        7. yanıtlarını gereksiz uzatma
        8. Müşteri sorarsa teknik detayları söyle
        

        YANIT:"""
        return prompt

    def get_response(self, question: str) -> Dict:
        """Get intelligent response to user question."""
        if not question.strip():
            return {
                "answer": "Merhaba! Size nasıl yardımcı olabilirim? Hizmetlerimiz, fiyatlarımız veya iletişim bilgilerimiz hakkında soru sorabilirsiniz.",
                "success": True,
                "sources": [],
                "confidence": 0.5
            }


        try:
            start_time = datetime.now()
            logger.info(f"❓ İşlenen soru: {question}")

            relevant_chunks = self.find_relevant_content(question)

            confidence = 0.3
            context_texts = [chunk for chunk, score in relevant_chunks]
            if relevant_chunks:
                confidence = min(0.95, max(0.3, relevant_chunks[0][1]))

            prompt = self._create_prompt(question, context_texts)

            # Get AI response
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 200,
                    'stop': ['\n\nMÜŞTERİ', '\n\nSORU', 'YANIT KURALLARI']
                }
            )

            answer = response['response'].strip()
            if answer.startswith('YANIT:'):
                answer = answer[6:].strip()

            # Clean up and append to history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Yanıt oluşturuldu ({processing_time:.2f}s, güven: {confidence:.2f})")

            return {
                "answer": answer,
                "success": True,
                "sources": [chunk[:100] + "..." for chunk, _ in relevant_chunks],
                "confidence": confidence,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"❌ Yanıt oluşturma hatası: {e}")
            return {
                "answer": "Üzgünüm, şu anda teknik bir sorun yaşıyorum. Lütfen daha sonra tekrar deneyin veya doğrudan bizimle iletişime geçin: info@techsoft.com.tr",
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }


def test_optimized_chatbot_engine():
    """Comprehensive test of the optimized chatbot engine."""
    print("🧪 Gelişmiş Türkçe Chatbot Engine Test Başlıyor...")
    print("=" * 60)

    try:
        # Initialize chatbot with optimized model and an embedding model
        # You need to have 'all-minilm' pulled with 'ollama pull all-minilm'
        chatbot = OptimizedTurkishChatbotEngine("data/knowledge_base.md", embedding_model="all-minilm")

        test_questions = [
            "Merhaba ",  # Tests greetings
            "Web siteleri için fiyat aralığınız nedir?",  # Tests pricing
            "Kurumsal web sitesi için ne kadar zaman ayırmalıyım?",  # Tests duration
            "Merhaba, size nasıl ulaşabilirim?",  # Tests contact
            "Şirketiniz neler yapıyor?",  # Tests general services
            "Ne tür teknoloji kullanıyorsunuz?",  # Tests tech stack
            "Yapay zeka projeleri için destek süresi ne kadar?",  # Tests specific support
            "Proje öncesi toplantılar ücretsiz mi?",  # Tests FAQ and specific info
            "Hangi veritabanlarını kullanıyorsunuz?",  # Tests tech stack
            "Web sitesi için ne kadar ödemem gerekir?",  # Tests pricing with different phrasing
            "Sizden hangi sektörlerde hizmet alabilirim?",  # Tests FAQ
            "Hangi ödeme yöntemlerini kabul ediyorsunuz?"  # Tests payment
        ]

        successful_responses = 0
        total_questions = len(test_questions)

        for i, question in enumerate(test_questions, 1):
            print(f"\n[{i}/{total_questions}] ❓ Soru: {question}")
            response = chatbot.get_response(question)

            if response["success"]:
                print(f"✅ Yanıt: {response['answer']}")
                print(f"📊 Güven Skoru: {response['confidence']:.2f}")
                print(f"⏱️ İşlem Süresi: {response.get('processing_time', 0):.2f}s")
                if response["sources"]:
                    print(f"📚 Kaynak Sayısı: {len(response['sources'])}")
                successful_responses += 1
            else:
                print(f"❌ Hata: {response.get('error', 'Bilinmeyen hata')}")
            print("-" * 50)

        success_rate = (successful_responses / total_questions) * 100
        print(f"\n🎯 TEST SONUÇLARI:")
        print(f"✅ Başarılı Yanıt: {successful_responses}/{total_questions}")
        print(f"📈 Başarı Oranı: {success_rate:.1f}%")

        if success_rate >= 90:
            print("\n🎉 Chatbot engine test başarıyla tamamlandı!")
            return True
        else:
            print("\n⚠️ Chatbot engine bazı testlerde başarısız oldu.")
            return False

    except Exception as e:
        print(f"❌ Test başarısız: {e}")
        return False


if __name__ == "__main__":
    test_optimized_chatbot_engine()