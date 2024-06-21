<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">
	<h1>여행계획</h1>
	<table class="table">
		<thead class="thead-light">
			<tr>
				<th>제목</th>
				<th>내용</th>
			</tr>
		</thead>
		<c:forEach var="gpt" items="${gpts.content}">
			<tbody>
				<tr>
					<td><a href="/user/planDetail/${gpt.id}" style="color: black;">${gpt.stage}</a></td>
					<td><a href="/user/planDetail/${gpt.id}" style="color: black;">${fn:substring(gpt.result, 0, 70)}</a></td>
				</tr>
			</tbody>
		</c:forEach>
	</table>

	<ul class="pagination justify-content-center">
		<c:choose>
			<c:when test="${gpts.first}">
				<li class="page-item disabled"><a class="page-link" href="?page=${gpts.number-1}">Previous</a></li>
			</c:when>
			<c:otherwise>
				<li class="page-item"><a class="page-link" href="?page=${gpts.number-1}">Previous</a></li>
			</c:otherwise>
		</c:choose>

		<!--  해당 페이지 순번 (이전, 다음 중간에 넣어주기) -->
		<c:forEach var="i" begin="1" end="${gpts.totalPages}">
			<li class="page-item"><a class="page-link" href="?page=${i-1}">${i}</a></li>
		</c:forEach>

		<c:choose>
			<c:when test="${gpts.last}">
				<!-- 만약 마지막 페이지면 disable -->
				<li class="page-item disabled"><a class="page-link" href="?page=${gpts.number+1}">Next</a></li>
			</c:when>
			<c:otherwise>
				<li class="page-item"><a class="page-link" href="?page=${gpts.number+1}">Next</a></li>
			</c:otherwise>
		</c:choose>
	</ul>
</div>

<script src="/js/user.js"></script>
<%@ include file="../layout/footer.jsp"%>