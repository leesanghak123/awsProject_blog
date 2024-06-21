<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="layout/header.jsp"%>

<div class="container">
	<h1>자유게시판</h1>
	<div class="text-right">
		<a href="/board/saveForm" class="btn btn-dark">글쓰기</a>
	</div>

	<table class="table">
		<thead class="thead-light">
			<tr>
				<th>순번</th>
				<th>제목</th>
				<th>글쓴이</th>
				<th>작성일</th>
				<th>댓글수</th>
				<th>조회수</th>
				<th>추천수</th>
			</tr>
		</thead>
		<c:forEach var="board" items="${boards.content}">
			<tbody>
				<tr>
					<td><a href="/board/${board.id}" style="color: black;">${board.id}</a></td>
					<td><a href="/board/${board.id}" style="color: black;">${fn:substring(board.title, 0, 10)}</a></td>
					<td>${fn:substring(board.user.username, 0, 8)}</td>
					<td><fmt:formatDate value="${board.createDate}" pattern="yyyy-MM-dd" /></td>
					<td>${board.replys.size()}</td>
					<td>${board.count}</td>
					<td><span class="badge badge-primary badge-pill">${board.good}</span></td>
				</tr>
			</tbody>
		</c:forEach>
	</table>
</div>

<ul class="pagination justify-content-center">
	<c:choose>
		<c:when test="${boards.first}">
			<li class="page-item disabled"><a class="page-link" href="?page=${boards.number-1}">Previous</a></li>
		</c:when>
		<c:otherwise>
			<li class="page-item"><a class="page-link" href="?page=${boards.number-1}">Previous</a></li>
		</c:otherwise>
	</c:choose>

	<c:forEach var="i" begin="1" end="${boards.totalPages}">
		<li class="page-item"><a class="page-link" href="?page=${i-1}">${i}</a></li>
	</c:forEach>

	<c:choose>
		<c:when test="${boards.last}">
			<li class="page-item disabled"><a class="page-link" href="?page=${boards.number+1}">Next</a></li>
		</c:when>
		<c:otherwise>
			<li class="page-item"><a class="page-link" href="?page=${boards.number+1}">Next</a></li>
		</c:otherwise>
	</c:choose>
</ul>
</div>

<style>
.text-right {
	text-align: right;
}

.text-right .btn {
	margin-bottom: 1vh;
}
</style>

<%@ include file="layout/footer.jsp"%>