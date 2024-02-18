from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:adminadmin@localhost:3306/blogapplication'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Blog(db.Model):
    blog_id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)
    blog_comment_count = db.Column(db.Integer)
    blog_user_user_id = db.Column(db.Integer, db.ForeignKey('blog_user.user_id'))

class BlogHistory(db.Model):
    blog_history_id = db.Column(db.Integer, primary_key=True)
    blog_user_user_id = db.Column(db.Integer, db.ForeignKey('blog_user.user_id'))
    blog_blog_id = db.Column(db.Integer, db.ForeignKey('blog.blog_id'))
    blog = db.relationship('Blog', backref=db.backref('histories', lazy=True))

def get_user_blogs(user_id):
    return Blog.query.join(BlogHistory, Blog.blog_id == BlogHistory.blog_blog_id).filter(BlogHistory.blog_user_user_id == user_id).all()

def calculate_user_similarity(user_id):
    all_user_ids = [user.blog_user_user_id for user in BlogHistory.query.with_entities(BlogHistory.blog_user_user_id).distinct().all()]
    current_user_blog_ids = set([blog.blog_id for blog in get_user_blogs(user_id)])
    
    similarities = []
    for other_user_id in all_user_ids:
        if other_user_id != user_id:
            other_user_blog_ids = set([blog.blog_id for blog in get_user_blogs(other_user_id)])
            jaccard_sim = len(current_user_blog_ids.intersection(other_user_blog_ids)) / len(current_user_blog_ids.union(other_user_blog_ids))
            similarities.append((other_user_id, jaccard_sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [user_id for user_id, sim in similarities[:3]]

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['userId']
    
    user_blogs = get_user_blogs(user_id)
    if not user_blogs:
        return jsonify([])

    all_blogs = Blog.query.all()
    user_blog_contents = [" ".join(blog.content.split()[:500]) for blog in user_blogs]
    all_blog_contents = [" ".join(blog.content.split()[:500]) for blog in all_blogs]

    vectorizer = TfidfVectorizer(stop_words='english')
    all_features = vectorizer.fit_transform(all_blog_contents)
    user_features = vectorizer.transform(user_blog_contents)

    cosine_similarities = cosine_similarity(user_features, all_features)
    total_similarities = np.sum(cosine_similarities, axis=0)
    top_50_indices = np.argsort(total_similarities)[::-1][:50]
    top_50_blog_ids = [all_blogs[i].blog_id for i in top_50_indices]

    similar_users = calculate_user_similarity(user_id)

    similar_users_blogs = Blog.query.join(BlogHistory, Blog.blog_id == BlogHistory.blog_blog_id).filter(BlogHistory.blog_user_user_id.in_(similar_users), Blog.blog_id.notin_([blog.blog_id for blog in user_blogs])).all()
    similar_users_blog_ids = [blog.blog_id for blog in similar_users_blogs]

    combined_blog_ids = list(set(top_50_blog_ids + similar_users_blog_ids))

    combined_blogs_scores = {blog_id: total_similarities[all_blogs.index(next(blog for blog in all_blogs if blog.blog_id == blog_id))] for blog_id in combined_blog_ids}
    sorted_combined_blog_ids = sorted(combined_blog_ids, key=lambda x: combined_blogs_scores[x], reverse=True)

    return jsonify(sorted_combined_blog_ids)


@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    user_query = data['text']
    if not user_query.strip():
        return jsonify([])  # 如果用户查询为空，则返回空列表

    keywords = [
    "Java",
    "Spring Framework",
    "Python",
    "JavaScript",
    "HTML",
    "CSS",
    "SQL",
    "C++",
    "Ruby",
    "PHP",
    "Machine Learning",
    "Artificial Intelligence",
    "Data Science",
    "Web Development",
    "DevOps",
    "Cloud Computing",
    "Docker",
    "Kubernetes",
    "Cybersecurity",
    "Network Security",
    "Big Data",
    "Blockchain",
    "Internet of Things (IoT)",
    "Mobile App Development",
    "Software Engineering",
    "Software Development",
    "Version Control (e.g., Git)",
    "Frontend Development",
    "Backend Development",
    "API (Application Programming Interface)",
    "Object-Oriented Programming (OOP)",
    "Agile Methodology",
    "Scrum",
    "Test-Driven Development (TDD)",
    "Continuous Integration (CI)",
    "Continuous Deployment (CD)",
    "Web Design",
    "User Experience (UX) Design",
    "Responsive Design",
    "Mobile Development",
    "Operating System",
    "Database Management",
    "NoSQL",
    "Machine Learning Algorithms",
    "Deep Learning",
    "Natural Language Processing (NLP)",
    "Computer Vision",
    "Cybersecurity Threats",
    "Penetration Testing", 
    "Network Protocols",
    "andriod"
    "C#"
    "gongxifacai"
]

    # 创建一个TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer()

    # 对关键词列表进行向量化
    tfidf_matrix = tfidf_vectorizer.fit_transform(keywords)

    # 分割用户查询中的每个词，并单独向量化
    query_words = user_query.split()
    query_matrix = tfidf_vectorizer.transform(query_words)

    # 计算每个词与关键词列表的余弦相似度
    cosine_similarities = cosine_similarity(query_matrix, tfidf_matrix)

    # 取每个关键词的最大相似度作为最终得分
    final_similarities = np.max(cosine_similarities, axis=0)

    # 获取相似度大于0.5的关键词的索引和相似度值
    similarities_with_index = [(index, sim) for index, sim in enumerate(final_similarities) if sim > 0.5]

    # 提取相似关键词
    similar_keywords = []
    if similarities_with_index:
        similarities_with_index.sort(key=lambda x: x[1], reverse=True)
        similar_keywords = [keywords[index] for index, _ in similarities_with_index[:3]]

    return jsonify(similar_keywords)
@app.route('/find_similar_blogs', methods=['POST'])
def find_similar_blogs():
    data = request.json
    user_query = data.get('text', '')

    # 检查用户查询是否提供
    if not user_query:
        return jsonify({"error": "No text provided"}), 400

    # 调用函数来找出与用户查询最相关的前50篇博客
    similar_blog_ids = find_similar_blogs_based_on_content(user_query)

    # 返回找到的博客ID列表
    return jsonify(similar_blog_ids)

# 之前定义的find_similar_blogs_based_on_content函数保持不变
def find_similar_blogs_based_on_content(user_query):
    all_blogs = Blog.query.all()
    all_blog_contents = [blog.content for blog in all_blogs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_blog_contents)
    query_vector = vectorizer.transform([user_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_50_indices = np.argsort(cosine_similarities)[-50:]
    top_50_blog_ids = [all_blogs[index].blog_id for index in reversed(top_50_indices)]
    return top_50_blog_ids


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
#curl -X POST http://localhost:8080/test-extraction -H "Content-Type: application/json" -d "{\"text\": \"i love spring java machine learning.\"}"
